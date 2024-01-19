# Multi-Dimension Full Scene Integrated Visual Emotion Analysis Network  
![network](network.png)
The Pytorch implementation of “Multi-Dimension Full Scene Integrated Visual Emotion Analysis Network”.

This is a preprint version for  2024 ICME.

## Requirements 

- Python
- torch
- torchvison
- pandas
- opencv
- seaborn
- numpy
- matplotlib

## Usage

All the code is in the folder ‘src’, you just need to decompress the file `ECANet.rar` in the same path, and run the `main.py`  file. Before running, you need to need to replace your model, data set and so on the correct path:
- Line36: load the pre-trained model (eca-renset101 with Imagenet)  
- Line40:  load the pre-trained model (resnet18 with FER2013)
- Line155: with your path of dateset (ImageNet-like)
- Line195: with your path to save model

Run it. Metric is Top1@Acc.

Our pretrained model is coming soon.

