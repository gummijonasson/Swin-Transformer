# Swin-Transformer
Implementation of [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

## Architecture
![Architecture of Swin Transformer.](figures/Swin%20Architecture.png)

## Usage
1. Open the folder "train_script".
2. Run the script for model training.

## Function of Each File
1. model/model.py: Define the architecture of the overall network.
2. model/patch_embed.py: Realize the image partition and patching embedding operations.
3. model/patch_merge.py: Realize the patch merging layer.
4. model/swin_block_noswin.py: Realize the swin Transformer block with W-MSA.
5. model/swin_block_noswin.py: Realize the swin Transformer block with SW-MSA.
6. train_script/train_base.ipynb: Train the base Swin Transformer.
7. train_script/train_base-no-shift.ipynb: Train the base Swin Transformer without window shifting.
8. train_script/base_tiny.ipynb: Train the tiny Swin transformer.
9. train_script/base_resnet.ipynb: Train the ResNet50.
10. train_script/base_mobilenet.ipynb: Train the MobileNet.

## Training Logs
[Google Drive](https://drive.google.com/drive/folders/1XuAZVsuUJ6ALUTjEK1jCmwNL_X0BWDrN?usp=share_link)

## Trained Models
[Google Drive](https://drive.google.com/drive/folders/1uCe0ZMivDMAltnwRQLE-gdI17trpqtlL?usp=share_link)

## Dataset
[Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html) is used in this project

## Directory Tree
```

├── LICENSE
├── README.md
├── figures
│   ├── Swin Architecture.png
│   ├── Training_acc.png
│   ├── training_loss.png
│   ├── val_acc.png
│   └── val_loss.png
├── model
│   ├── __init__.py
│   ├── model.py
│   ├── patch_embed.py
│   ├── patch_merge.py
│   ├── swin_block_noswin.py
│   └── swin_block_swin.py
└── train_script
    ├── train_base-no-shift.ipynb
    ├── train_base.ipynb
    ├── train_mobilenet.ipynb
    ├── train_resnet.ipynb
    └── train_tiny.ipynb
└── E4040.2022Fall.WWWW.report.gmj2122.vk2496.wg2397
```

