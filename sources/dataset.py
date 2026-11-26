import os
from torchvision import transforms, datasets
import torch.utils.data as data
import torch


def create_loader(batch_size, data_dir, data):
    data_dir = os.path.join(data_dir, data)
    if data == 'CIFAR100':
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),])

        trainset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        num_classes = 100
        image_size = 32

        return train_loader, test_loader, num_classes, image_size
