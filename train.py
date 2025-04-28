import gc
import os
import sys
import time
import dist
import copy
import yaml
import shutil
import warnings
from functools import partial
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume

from models.server import FederatedServer
from models.client import FederatedClient

def load_data_config(config_path: str = 'configs/data_config.yaml') -> Dict:
    """Load data configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def build_everything(args: arg_util.Args):
    """Build all components for training."""
    # Auto resume setup
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    
    
    # Create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()
    
    # Log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    
    # Initialize federated setup
    # Load data configuration
    data_config = load_data_config()

    if args.case is not None:
        case = args.case
    else:
        case = data_config['data']['case']

    clients = data_config['data']['clients'][case]
    client_num = len(clients)  # Get actual number of clients from config
    assert client_num == args.client_num, f"Mismatch: {len(clients)} clients created but args.client_num={args.client_num}"
    ld_train_dict = {}
    ld_val_dict = {}
    iters_train_dict = {}

    # Build data loaders for each client
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        stt = time.time()
        for client_id in range(client_num):
            print(f'[build PT data for client {client_id}] ...\n')
            
            # Get data path from config
            client_config = clients[client_id]
            args.data_path = client_config['path']
            print(f'Using data path: {args.data_path}')
            
            # Build datasets
            num_classes, dataset_train, dataset_val = build_dataset(
                args.data_path,
                final_reso=args.data_load_reso,
                hflip=args.hflip,
                mid_reso=args.mid_reso,
            )
            types = str((type(dataset_train).__name__, type(dataset_val).__name__))
            # Create validation data loader
            ld_val = DataLoader(
                dataset_val,
                num_workers=0,
                pin_memory=True,
                batch_size=round(args.batch_size * 1.5),
                sampler=EvalDistributedSampler(
                    dataset_val,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank()
                ),
                shuffle=False,
                drop_last=False,
            )
            ld_val_dict[f'ld_val_{client_id}'] = ld_val
            del dataset_val
            
            # Create training data loader
            ld_train = DataLoader(
                dataset=dataset_train,
                num_workers=args.workers,
                pin_memory=True,
                generator=args.get_different_generator_for_each_rank(),
                batch_sampler=DistInfiniteBatchSampler(
                    dataset_len=len(dataset_train),
                    glb_batch_size=args.glb_batch_size,
                    same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                    shuffle=True,
                    fill_last=True,
                    rank=dist.get_rank(),
                    world_size=dist.get_world_size(),
                    start_ep=start_ep,
                    start_it=start_it,
                ),
            )
            del dataset_train
            
            iters_train_dict[f'iters_train_{client_id}'] = len(ld_train)
            ld_train_dict[f'ld_train_{client_id}'] = iter(ld_train)
            
            print(f'client: {client_id} - {client_config["name"]}')
            
            print(f'[dataloader multi processing] ...', end='', flush=True)
            print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train=iters_train_{client_id}')
        
        print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True)

 
    # Build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import GAT, VQVAE, create_vqvae_and_gat
    from trainer import GATTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params
    
    # Build VAE and GAT models
    vae_local, gat_wo_ddp = create_vqvae_and_gat(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,       
        device=dist.get_device(), patch_nums=args.patch_nums,
        num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    
    # Load checkpoints
    vae_ckpt = 'vae_ch160v4096z32.pth'
    #gat_ckpt = 'var_d16.pth'

    
    if dist.get_rank() == 0:
        if not os.path.exists(vae_ckpt):
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
        # if not os.path.exists(gat_ckpt):
        #     raise FileNotFoundError(f"The specified GAT model checkpoint '{gat_ckpt}' does not exist.")
    dist.barrier()
    
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    #gat_wo_ddp.load_state_dict(torch.load(gat_ckpt, map_location='cpu'), strict=False)

    # Compile models
    vae_local = args.compile_model(vae_local, args.vfast)
    gat_wo_ddp = args.compile_model(gat_wo_ddp, args.tfast)
    gat = DDP(gat_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)
    
    # Print model info
    print(f'[INIT] GAT model = {gat_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('GAT', gat_wo_ddp),)]) + '\n\n')
    
    # Build optimizer
    names, paras, para_groups = filter_params(gat_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    gat_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups
    
    # Build trainer
    trainer = GATTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, gat_wo_ddp=gat_wo_ddp, gat=gat,
        gat_opt=gat_optim, label_smooth=args.ls,
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True)
    del vae_local, gat_wo_ddp, gat, gat_optim
    
    
    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train_dict, ld_train_dict, ld_val_dict
    )

def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)

    tb_lg, trainer, start_ep, start_it, iters_train_dict, ld_train_dict, ld_val_dict = build_everything(args)

    # Get actual number of clients from config
    server = FederatedServer(trainer, args.device, tb_lg)
    client_num = args.client_num
    comm_round = args.comm_round
    clients = [FederatedClient(i, trainer,args, args.device, tb_lg) for i in range(client_num)]

    for client in clients:
        client.set_data(
            train_loader=ld_train_dict[f'ld_train_{client.client_id}'],
            val_loader=ld_val_dict[f'ld_val_{client.client_id}']
        )

    start_time = time.time()

    for ep in range(start_ep, args.ep):
        if (ep - start_ep) % comm_round == 0:
            # snapshot global model only every comm_round
            global_model_state = copy.deepcopy(server.trainer.gat_wo_ddp.state_dict())
        
        client_models = []
        client_metrics = []

        for client in clients:
            client.load_model(global_model_state)

            if hasattr(client.train_loader, 'sampler') and hasattr(client.train_loader.sampler, 'set_epoch'):
                client.train_loader.sampler.set_epoch(ep)

            stats = client.train_one_epoch(ep)
            client_models.append(client.get_model())
            client_metrics.append(stats)

        # Aggregate only every comm_round epochs
        if (ep - start_ep + 1) % comm_round == 0 or (ep + 1) == args.ep:
            avg_model = server.update(client_models, client_metrics, ep)

            if dist.is_local_master():
                ckpt_path = os.path.join(args.local_out_dir_path, f'ar-ckpt-avg-ep{ep+1}.pth')
                torch.save(avg_model, ckpt_path)
                print(f'[*] Saved averaged model at {ckpt_path}', flush=True)

        dist.barrier()

    total_time = f'{(time.time() - start_time) / 3600:.1f}h'
    print(f'[*] Training completed in {total_time}')

    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    args.dump_log()
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()


if __name__ == '__main__':
    try: 
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
            
            
