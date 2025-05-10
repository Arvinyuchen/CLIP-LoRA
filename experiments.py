import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import clip
from datasets import build_dataset
from datasets.utils import build_data_loader

from utils import *
from run_utils import *
from lora import run_lora


def main(shot: int) -> float:

    # Load config file
    args = get_arguments()
    
    set_random_seed(args.seed)
    
    # CLIP
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    logit_scale = 100

    # Prepare dataset
    print("Preparing dataset.")
        
    dataset = build_dataset(args.dataset, args.root_path, shot, preprocess)
    
    if args.dataset == 'imagenet':
        val_loader = torch.utils.data.DataLoader(dataset.val, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=256, num_workers=8, shuffle=False, pin_memory=True)
    else:
        val_loader = build_data_loader(data_source=dataset.val, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        test_loader = build_data_loader(data_source=dataset.test, batch_size=256, is_train=False, tfm=preprocess, shuffle=False,  num_workers=8)
        
    train_loader = None
    if not args.eval_only:
        train_tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.08, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        if args.dataset == 'imagenet':
            train_loader = torch.utils.data.DataLoader(dataset.train_x, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        else:
            train_loader = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size, tfm=train_tranform, is_train=True, shuffle=True, num_workers=8)

    acc, zs_acc = run_lora(args, clip_model, logit_scale, dataset, train_loader, val_loader, test_loader)
    return acc, zs_acc

def run_experiments():
    shots = [1]
    results = {}
    zs_acc = None
    

    for shot in shots:
        acc, shot_zs_acc = main(shot)
        if zs_acc is None:
            zs_acc = shot_zs_acc
        results[shot] = acc


    # Plotting zero-shot accuracy as star
    plt.scatter([0], [zs_acc], color='mediumorchid', marker='*', s=200, label='Zero-Shot\nCLIP')
    plt.text(0, zs_acc - 1.5, "Zero-Shot\nCLIP", color='mediumorchid', ha='center', fontsize=10)

    # Plotting few -shots 
    plt.plot(list(results.keys()), list(results.values()), color='mediumorchid', marker='o', linestyle=':')
    plt.xlabel("# of labeled training examples per class", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    run_experiments()
    