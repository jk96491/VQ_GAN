import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn


def MnistLoadData(image_size, batch_size, train, generate_image):
    if generate_image is True:
        os.makedirs("images", exist_ok=True)

    if image_size is None:
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    os.makedirs("../../Data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../Data/mnist",
            train=train,
            download=True,
            transform=transform),
        batch_size=batch_size,
        shuffle=True,
    )

    return dataloader


def CIFARLoadData(batch_size, Train, generate_image):
    if generate_image is True:
        os.makedirs("images", exist_ok=True)

    transform = transforms.Compose([transforms.Scale(64), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root='../../Data/CIFAR/', train=Train, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


class TrainImageFolder(datasets.ImageFolder):

    def __getitem__(self, index):
        filename = self.imgs[index]
        if filename[0].split('\\')[2] == 'out_dry01':
            label = 0
        elif filename[0].split('\\')[2] == 'out_normal01':
            label = 1
        else:
            label = 2

        return super(TrainImageFolder, self).__getitem__(index)[0], label


def SkinDataLoad(batch_size, generate_image):
    if generate_image is True:
        os.makedirs("images", exist_ok=True)
        os.makedirs("images/dry", exist_ok=True)
        os.makedirs("images/normal", exist_ok=True)
        os.makedirs("images/wet", exist_ok=True)

    traindir = 'skinData\\Train'
    train_loader = torch.utils.data.DataLoader(
        TrainImageFolder(traindir,
                         transforms.Compose([
                             transforms.RandomResizedCrop(64),
                             transforms.RandomHorizontalFlip(),
                             #transforms.Grayscale(num_output_channels=1),
                             transforms.ToTensor(),
                             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                             # normalize,
                         ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    return  train_loader


def conv_3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv_1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_device(device_name='cuda:0'):
    device = device_name if torch.cuda.is_available() else 'cpu'
    return device