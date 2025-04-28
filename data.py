import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
import random

class RotatedCIFAR10(CIFAR10):
    def __init__(self, root, train=True, download=True, transform=None, rotation_angle=None):
        super().__init__(root, train=train, download=download)
        self.rotation_angle = rotation_angle
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        # Random rotation if no fixed angle specified
        angle = self.rotation_angle if self.rotation_angle is not None else random.uniform(0, 360)
        img = transforms.functional.rotate(img, angle)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

