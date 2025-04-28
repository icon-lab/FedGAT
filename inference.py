import os
import os.path as osp
import random
import numpy as np
import torch
import torchvision
import PIL.Image as PImage

from models import VQVAE, create_vqvae_and_gat

def prepare_model(vae_ckpt_path, gat_ckpt_path, device, model_depth):
    """Load VQVAE and GAT models."""
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    vae, gat = create_vqvae_and_gat(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=model_depth,
        shared_aln=False,
    )

    vae.load_state_dict(torch.load(vae_ckpt_path, map_location='cpu'), strict=True)
    checkpoint = torch.load(gat_ckpt_path, map_location='cpu')
    gat.load_state_dict(checkpoint, strict=False)

    vae.eval()
    gat.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in gat.parameters(): p.requires_grad_(False)

    return vae, gat


def set_seed(seed):
    """Ensure reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_images(gat, output_dir, total_classes=3, images_per_class=840, batch_size=1, cfg=1.5, device='cuda'):
    """Generate synthetic images and save one .npy file per class."""
    os.makedirs(output_dir, exist_ok=True)

    for class_id in range(total_classes):
        print(f"[*] Generating images for class {class_id}...")
        all_images = []  # Collect all images for this class
        img_count = 0

        while img_count < images_per_class:
            current_batch_size = min(batch_size, images_per_class - img_count)
            class_labels = torch.full((current_batch_size,), class_id, dtype=torch.long, device=device)

            recon_B3HW = []
            for idx in range(current_batch_size):
                seed = img_count + idx
                set_seed(seed)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    recon = gat.autoregressive_infer_cfg(
                        B=1, label_B=class_labels[idx:idx+1],
                        cfg=cfg, top_k=900, top_p=0.95,
                        g_seed=seed, more_smooth=False
                    )
                recon_B3HW.append(recon)

            for recon in recon_B3HW:
                img_array = recon[0].cpu().numpy()  # (C, H, W)
                all_images.append(img_array)

            img_count += current_batch_size

        # Stack and save for this class
        all_images = np.stack(all_images, axis=0)  # (images_per_class, C, H, W)
        save_path = osp.join(output_dir, f'class_{class_id}.npy')
        np.save(save_path, all_images)
        print(f'[*] Saved {images_per_class} images for class {class_id} to {save_path}. Shape: {all_images.shape}')


def main():
    ##################################
    # Configurations
    ##################################
    ep = 'ep500'  # Epoch identifier
    model_depth = 16
    num_images_per_class = 840
    batch_size = 1
    total_classes = 3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_scale = 1.5  # Classifier-free guidance

    base_dir = '/auto/k2/ansarian/Desktop/FL/VAR2/VAR/VAR_ablations/gatedmlp/models'
    vae_ckpt_path = '/auto/k2/ansarian/Desktop/FL/VAR2/VAR/vae_ch160v4096z32.pth'

    gat_ckpt_path = '/auto/k2/ansarian/Desktop/FL/VAR2/VAR/VAR_ablations/gatedmlp/fedGAT_output/ar-ckpt-avg-{ep}.pth'
    output_dir = osp.join(base_dir, ep)

    ##################################
    # Prepare models
    ##################################
    print('[*] Preparing models...')
    vae, gat = prepare_model(vae_ckpt_path, gat_ckpt_path, device, model_depth)
    print('[*] Model preparation complete.')

    ##################################
    # Generate and save images
    ##################################
    print('[*] Starting image generation...')
    generate_images(
        gat=gat,
        output_dir=output_dir,
        total_classes=total_classes,
        images_per_class=num_images_per_class,
        batch_size=batch_size,
        cfg=cfg_scale,
        device=device
    )
    print(f'[*] All images saved to {output_dir}')


if __name__ == '__main__':
    main()
