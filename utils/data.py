import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


import os.path as osp
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader



def pil_loader(path):
    from PIL import Image
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# Function to build dataset
def build_dataset(data_path: str, final_reso: int, hflip=False, mid_reso=1.125):
    # Build augmentations
    mid_reso = round(mid_reso * final_reso)  # First resize to mid_reso, then crop to final_reso
    train_aug = [
        transforms.ToTensor(),
        transforms.Lambda(normalize_01_into_pm1),  # Apply normalization
    ]
    
    val_aug = [
        transforms.ToTensor(),
        transforms.Lambda(normalize_01_into_pm1),  # Apply normalization
    ]
    
    train_aug = transforms.Compose(train_aug)
    val_aug = transforms.Compose(val_aug)
    
    # Build dataset
    train_set = DatasetFolder(root=osp.join(data_path, 'train'), loader=pil_loader, extensions=('jpg', 'jpeg', 'png'), transform=train_aug)
    val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=('jpg', 'jpeg', 'png'), transform=val_aug)
    num_classes = 1000
    
    print(f'[Dataset] len(train_set)={len(train_set)}, len(val_set)={len(val_set)}, num_classes={num_classes}')
    print_aug(train_aug, '[train]')
    print_aug(val_aug, '[val]')
    
    return num_classes, train_set, val_set



# Function to print augmentations
def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')

