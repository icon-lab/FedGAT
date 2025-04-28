from trainer import GATTrainer
import warnings
import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple
import dist
from utils import arg_util, misc
import copy

from utils.misc import TensorboardLogger # for TensorboardLogger and SyncPrint

class FederatedClient:
    """Per-client training of GAT model in a federated setup."""
    def __init__(
        self,
        client_id: int,
        trainer,
        args,
        device: torch.device,
        logger: misc.TensorboardLogger
    ):
        self.client_id = client_id
        self.trainer = trainer
        self.args = args
        self.device = device
        self.logger = logger
        self.train_loader = None
        self.val_loader = None
        # Optional: assign network address if provided
        self.address = None
        if hasattr(args, 'client_addresses'):
            self.address = args.client_addresses[client_id]

    def set_data(self, train_loader, val_loader):
        """Set the training and validation data loaders."""
        self.train_loader = train_loader
        self.val_loader = val_loader

    def load_model(self, state_dict: Dict):
        """Load global weights into the local model."""
        self.trainer.gat_wo_ddp.load_state_dict(state_dict)
    
    def train_one_epoch(self, ep: int) -> Dict:
        """Train one epoch on client data with proper logger steps."""
        addr_info = f" at {self.address}" if self.address else ""
        print(f"Training model for client {self.client_id}{addr_info} at epoch {ep}...")

        # set TB logger step consistent with main_training
        step = ep * len(self.train_loader)
        self.logger.set_step(step)

        stats, (sec, remain_time, finish_time) = self.train_one_ep(
            ep=ep,
            is_first_ep=(ep == 0),
            start_it=0,
            args=self.args,
            tb_lg=self.logger,
            ld_or_itrt=self.train_loader,
            iters_train=len(self.train_loader),
            client_id=self.client_id,
            trainer=self.trainer
        )

        # Log metrics under 'AR_ep_loss'
        self.logger.update(
            head='AR_ep_loss',
            step=ep + 1,
            L_mean=stats['Lm'],
            L_tail=stats['Lt'],
            acc_mean=stats['Accm'],
            acc_tail=stats['Acct']
        )
        # Log additional metrics if needed
        self.logger.update(
            head='AR_z_burnout',
            step=ep + 1,
            rest_hours=round(sec / 3600, 2)
        )

        return stats

    def get_model(self) -> Dict:
        """Get a copy of the local model's state dict for aggregation."""
        return copy.deepcopy(self.trainer.gat_wo_ddp.state_dict())

    def validate(self) -> Dict:
        """Run validation on the client's validation set."""
        self.trainer.gat_wo_ddp.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for inp, label in self.val_loader:
                inp = inp.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                outputs = self.trainer.forward(inp, label)
                total_loss += outputs['loss'].item()
                num_batches += 1
        return {'val_loss': total_loss / num_batches if num_batches > 0 else 0.0}
    @staticmethod
    def train_one_ep(
        ep: int,
        is_first_ep: bool,
        start_it: int,
        args: arg_util.Args,
        tb_lg: misc.TensorboardLogger,
        ld_or_itrt,
        iters_train: int,
        client_id: int,
        trainer,
    ):
        from trainer import GATTrainer
        from utils.lr_control import lr_wd_annealing
        import torchvision.utils as vutils
        trainer: GATTrainer
        
        step_cnt = 0
        me = misc.MetricLogger(delimiter='  ')

        me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
        me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
        [me.add_meter(x, misc.SmoothedValue(fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]

        header = f'[Ep]: [{ep:4d}/{args.ep}]'
        
        if is_first_ep:
            warnings.filterwarnings('ignore', category=DeprecationWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
        g_it, max_it = ep * iters_train, args.ep * iters_train


        
        for it, (inp, label) in me.log_every(start_it, iters_train, ld_or_itrt, 30 if iters_train > 8000 else 5, header):

            g_it = ep * iters_train + it
            if it < start_it: continue
            if is_first_ep and it == start_it: warnings.resetwarnings()

            inp = inp.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)
            
            # Replace labels with the client number
            label = torch.full_like(label, client_id, dtype=torch.long, device=args.device)
            
            args.cur_it = f'{it+1}/{iters_train}'
            
            wp_it = args.wp * iters_train
            min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.gat_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
            args.cur_lr, args.cur_wd = max_tlr, max_twd
            
            if args.pg: # default: args.pg == 0.0, means no progressive training, won't get into this
                if g_it <= wp_it: prog_si = args.pg0
                elif g_it >= max_it*args.pg: prog_si = len(args.patch_nums) - 1
                else:
                    delta = len(args.patch_nums) - 1 - args.pg0
                    progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1) # from 0 to 1
                    prog_si = args.pg0 + round(progress * delta)    # from args.pg0 to len(args.patch_nums)-1
            else:
                prog_si = -1
            
            stepping = (g_it + 1) % args.ac == 0
            step_cnt += int(stepping)
            print('it',it, 'g_it',g_it, 'stepping',stepping, 'metric_lg',me, 'tb_lg',tb_lg,
                'label_B',label, 'prog_si',prog_si, 'prog_wp_it',args.pgwp * iters_train)
            
            grad_norm, scale_log2 = trainer.train_step(
                it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
                inp_B3HW=inp, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
            )
            
            me.update(tlr=max_tlr)
            tb_lg.set_step(step=g_it)
            tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
            tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
            tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
            tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
            tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
            
            if args.tclip > 0:
                tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
                tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
            
        
        # # Log an example image at the end of each epoch
        # # Log an example image at the end of each epoch
        # #example_image = inp[0].cpu()  # Get the first image in the batch and move it to CPU
        # #print(f'[train_one_ep] Logging example image at epoch {ep}')  # Debug print
        # #tb_lg.log_image('Example Image', example_image, step=ep)
        # example_image = inp[0].cpu()  # Get the first image in the batch and move it to CPU
                    
        # example_image = (example_image + 1) / 2  # Convert [-1, 1] to [0, 1]
        #             # Convert to numpy array
        # example_image_np = example_image.numpy()
        # print(f'[train_step] Logging example image at step {ep}')  # Debug print
        # tb_lg.log_image('Example Image', example_image_np, step=ep)
        # print('min:',example_image.min)
        # print('max:',example_image.max)
        me.synchronize_between_processes()
        return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost

