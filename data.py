import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100, CelebA
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


# --- CIFAR 100 --- 
class CIFAR100RotationTestDataset(Dataset):
    def __init__(self, root, rotation_range=(0, 360), train=False):
        """
        Args:
            root (str): Path to dataset.
            rotation_range (tuple): (min_angle, max_angle) for random rotation.
            train (bool): Load training split if True, else test split.
        """
        self.dataset = CIFAR100(
            root=root,
            train=train,
            download=True,
            transform=None 
        )
        self.rotation_min, self.rotation_max = rotation_range

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Convert to tensor first (before rotation)
        img = TF.to_tensor(img)
        
        # Apply padding before rotation to avoid information loss
        # Calculate diagonal length to ensure rotated image fits
        diagonal = int(np.ceil(np.sqrt(2) * max(img.shape[1:])))
        pad_size = (diagonal - img.shape[1]) // 2
        img = TF.pad(img, padding=[pad_size, pad_size, pad_size, pad_size], padding_mode='reflect')
        
        # Apply random continuous rotation
        angle = random.uniform(self.rotation_min, self.rotation_max)
        img = TF.rotate(img, angle, expand=False)  # No expand to maintain size
        
        # Center crop to get back to desired size
        img = TF.center_crop(img, (32, 32))

        return img, label, angle

class CIFAR100TrainDataset(Dataset):
    def __init__(self, root):
        self.dataset = CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=None
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # NO rotation!
        img = TF.resize(img, (32, 32))
        img = TF.to_tensor(img)

        return img, label

def get_cifar100_train_loader(root, batch_size=128):
    dataset = CIFAR100TrainDataset(root=root)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader

def get_cifar100_rotation_test_loader(root, batch_size=128, rotation_range=(0, 360)):
    dataset = CIFAR100RotationTestDataset(root=root, rotation_range=rotation_range, train=False)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader


# --- CelebA ---
class CelebARotationTestDataset(Dataset):
    def __init__(self, root, split="test", rotation_range=(0, 360), target_type="attr", crop_bbox=False):
        self.dataset = CelebA(
            root=root,
            split=split,
            target_type=target_type,
            download=True,
            transform=None,
        )
        self.rotation_min, self.rotation_max = rotation_range
        self.crop_bbox = crop_bbox

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        if self.crop_bbox:
            # Optional: Crop to bounding box
            _, bbox, _ = self.dataset[idx][1] if isinstance(target, tuple) else (None, None, None)
            if bbox is not None:
                x, y, w, h = bbox
                img = TF.crop(img, top=int(y), left=int(x), height=int(h), width=int(w))

        img = TF.to_tensor(img)
        
        img = TF.resize(img, (128, 128))
        
        # Apply padding before rotation to avoid information loss
        # Calculate diagonal length to ensure rotated image fits
        diagonal = int(np.ceil(np.sqrt(2) * max(img.shape[1:])))
        pad_size = (diagonal - img.shape[1]) // 2
        img = TF.pad(img, padding=[pad_size, pad_size, pad_size, pad_size], padding_mode='reflect')
        
        angle = random.uniform(self.rotation_min, self.rotation_max)
        img = TF.rotate(img, angle, expand=False)  # No expand to maintain size
        
        img = TF.center_crop(img, (128, 128))

        return img, target, angle
    
class CelebATrainDataset(Dataset):
    def __init__(self, root, split="train", target_type="attr", crop_bbox=False):
        self.dataset = CelebA(
            root=root,
            split=split,
            target_type=target_type,
            download=True,
            transform=None,
        )
        self.crop_bbox = crop_bbox

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]

        if self.crop_bbox:
            _, bbox, _ = self.dataset[idx][1] if isinstance(target, tuple) else (None, None, None)
            if bbox is not None:
                x, y, w, h = bbox
                img = TF.crop(img, top=int(y), left=int(x), height=int(h), width=int(w))

        # NO rotation!
        img = TF.resize(img, (128, 128))
        img = TF.to_tensor(img)

        return img, target

def get_celeba_train_loader(root, batch_size=64):
    dataset = CelebATrainDataset(
        root=root,
        split="train",
        target_type="attr",
        crop_bbox=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader

def get_celeba_rotation_test_loader(root, batch_size=64, rotation_range=(0, 360)):
    dataset = CelebARotationTestDataset(
        root=root,
        split="test",
        rotation_range=rotation_range,
        target_type="attr",
        crop_bbox=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader


if __name__ == "__main__":
    # Set paths to your datasets
    cifar_root = "./data/cifar-100-python"
    celeba_root = "./data/celeba"
    
    # Function to display images
    def show_images(images, title, angles=None):
        grid = make_grid(images, nrow=4, normalize=True)
        plt.figure(figsize=(12, 8))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(title)
        
        if angles is not None:
            # Convert tensor to list of floats before rounding
            angles_list = angles.tolist() if isinstance(angles, torch.Tensor) else angles
            plt.figtext(0.5, 0.01, f"Rotation angles: {[round(a, 1) for a in angles_list]}", 
                       ha='center', fontsize=10)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Visualize CIFAR100 images
    print("Loading CIFAR100 images...")
    cifar_train_loader = get_cifar100_train_loader(cifar_root, batch_size=8)
    cifar_test_loader = get_cifar100_rotation_test_loader(cifar_root, batch_size=8, 
                                                         rotation_range=(0, 360))
    
    # Get a batch of training images
    train_images, train_labels = next(iter(cifar_train_loader))
    show_images(train_images, "CIFAR100 Training Images (No Rotation)")
    
    # Get a batch of test images with rotation
    test_images, test_labels, test_angles = next(iter(cifar_test_loader))
    show_images(test_images, "CIFAR100 Test Images (With Random Rotation)", test_angles)
    
    # Visualize CelebA images
    print("Loading CelebA images...")
    try:
        celeba_train_loader = get_celeba_train_loader(celeba_root, batch_size=8)
        celeba_test_loader = get_celeba_rotation_test_loader(celeba_root, batch_size=8, 
                                                     rotation_range=(0, 360))
        
        # Get a batch of training images
        train_images, _ = next(iter(celeba_train_loader))
        show_images(train_images, "CelebA Training Images (No Rotation)")
        
        # Get a batch of test images with rotation
        test_images, _, test_angles = next(iter(celeba_test_loader))
        show_images(test_images, "CelebA Test Images (With Random Rotation)", test_angles)
    except Exception as e:
        print(f"Error loading CelebA dataset: {e}")
        print("Note: CelebA dataset is large and may require download first.")