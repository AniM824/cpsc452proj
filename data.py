import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image


# --- EMNIST ---

EMNIST_MEAN = [0.1307] * 3
EMNIST_STD  = [0.3081] * 3

class EMNISTBase(Dataset):
    def __init__(self, root, train, split='byclass', download=True):
        self.dataset = EMNIST(root=root, split=split, train=train, download=download, transform=None)
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def _preprocess(self, pil_img):
        img = TF.resize(pil_img, (32, 32))
        img_tensor = TF.to_tensor(img)
        return img_tensor.repeat(3, 1, 1)

    def _normalize(self, tensor_img):
        return TF.normalize(tensor_img, EMNIST_MEAN, EMNIST_STD)

class EMNISTTrainDataset(EMNISTBase):
    def __init__(self, root, split='byclass'):
        super().__init__(root, split=split, train=True)
        self.augmentations = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        ])

    def __getitem__(self, idx):
        pil_img, label = self.dataset[idx]
        img = self._preprocess(pil_img)
        img = self.augmentations(img)
        img = self._normalize(img)
        return img, label

class EMNISTRotationTestDataset(EMNISTBase):
    def __init__(self, root, split='byclass', rotation_range=(0, 360), apply_rotation=True):
        super().__init__(root, split=split, train=False)
        self.min_angle, self.max_angle = rotation_range
        self.apply_rotation = apply_rotation

    def __getitem__(self, idx):
        pil_img, label = self.dataset[idx]

        img = self._preprocess(pil_img)

        angle = 0.0
        if self.apply_rotation:
            diag = int(np.ceil(np.sqrt(2) * 32))
            pad = (diag - 32) // 2
            img = TF.pad(img, [pad] * 4, padding_mode='reflect')

            angle = random.uniform(self.min_angle, self.max_angle)
            img = TF.rotate(img, angle, expand=False)

            img = TF.center_crop(img, (32, 32))

        img = self._normalize(img)

        return img, label, angle

def get_emnist_train_loader(root, split='byclass', batch_size=128):
    dataset = EMNISTTrainDataset(root=root, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader

def get_emnist_rotation_test_loader(root, split='byclass', batch_size=128, rotation_range=(0, 360)):
    dataset = EMNISTRotationTestDataset(root=root, split=split, rotation_range=rotation_range, apply_rotation=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader

def get_emnist_test_loader(root, split='byclass', batch_size=128, rotation_range=(0, 360)):
    dataset = EMNISTRotationTestDataset(root=root, split=split, rotation_range=rotation_range, apply_rotation=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader


# --- CIFAR 10 --- 

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD  = [0.2023, 0.1994, 0.2010]

class CIFAR10Base(Dataset):
    def __init__(self, root, train, download=True):
        self.dataset = CIFAR10(root=root, train=train, download=download, transform=None)
    
    def __len__(self):
        return len(self.dataset)

    def _preprocess(self, pil_img):
        img = TF.resize(pil_img, (32, 32))
        return TF.to_tensor(img)

    def _normalize(self, tensor_img):
        return TF.normalize(tensor_img, CIFAR10_MEAN, CIFAR10_STD)

class CIFAR10TrainDataset(CIFAR10Base):
    def __init__(self, root):
        super().__init__(root, train=True)
        self.augmentations = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.1),
        ])

    def __getitem__(self, idx):
        pil_img, label = self.dataset[idx]
        img = self._preprocess(pil_img)
        img = self.augmentations(img)
        img = self._normalize(img)
        return img, label

class CIFAR10RotationTestDataset(CIFAR10Base):
    def __init__(self, root, rotation_range=(0, 360), apply_rotation=True):
        super().__init__(root, train=False)
        self.min_angle, self.max_angle = rotation_range
        self.apply_rotation = apply_rotation

    def __getitem__(self, idx):
        pil_img, label = self.dataset[idx]

        img = self._preprocess(pil_img)

        angle = 0.0 
        if self.apply_rotation:
            diag = int(np.ceil(np.sqrt(2) * 32))
            pad = (diag - 32) // 2
            img = TF.pad(img, [pad] * 4, padding_mode='reflect')

            angle = random.uniform(self.min_angle, self.max_angle)
            img = TF.rotate(img, angle, expand=False)

            img = TF.center_crop(img, (32, 32))

        img = self._normalize(img)

        return img, label, angle

def get_cifar10_train_loader(root, batch_size=128):
    dataset = CIFAR10TrainDataset(root=root)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader

def get_cifar10_rotation_test_loader(root, batch_size=128, rotation_range=(0, 360)):
    dataset = CIFAR10RotationTestDataset(root=root, rotation_range=rotation_range, apply_rotation=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader

def get_cifar10_test_loader(root, batch_size=128, rotation_range=(0, 360)):
    dataset = CIFAR10RotationTestDataset(root=root, rotation_range=rotation_range, apply_rotation=False)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader


# --- CIFAR 100 --- 

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD  = [0.2675, 0.2565, 0.2761]

class CIFAR100Base(Dataset):
    def __init__(self, root, train, download=True):
        self.dataset = CIFAR100(root=root, train=train, download=download, transform=None)

    def __len__(self):
        return len(self.dataset)

    def _preprocess(self, pil_img):
        img = TF.resize(pil_img, (32, 32))
        return TF.to_tensor(img)

    def _normalize(self, tensor_img):
        return TF.normalize(tensor_img, CIFAR100_MEAN, CIFAR100_STD)

class CIFAR100TrainDataset(CIFAR100Base):
    def __init__(self, root):
        super().__init__(root, train=True)
        self.augmentations = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.1),
        ])

    def __getitem__(self, idx):
        pil_img, label = self.dataset[idx]
        img = self._preprocess(pil_img)
        img = self.augmentations(img)
        img = self._normalize(img)
        return img, label

class CIFAR100RotationTestDataset(CIFAR100Base):
    def __init__(self, root, rotation_range=(0, 360), apply_rotation=True):
        super().__init__(root, train=False)
        self.min_angle, self.max_angle = rotation_range
        self.apply_rotation = apply_rotation

    def __getitem__(self, idx):
        pil_img, label = self.dataset[idx]

        img = self._preprocess(pil_img)

        angle = 0.0 
        if self.apply_rotation:
            diag = int(np.ceil(np.sqrt(2) * 32))
            pad = (diag - 32) // 2
            img = TF.pad(img, [pad] * 4, padding_mode='reflect')

            angle = random.uniform(self.min_angle, self.max_angle)
            img = TF.rotate(img, angle, expand=False)

            img = TF.center_crop(img, (32, 32))

        img = self._normalize(img)

        return img, label, angle

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
    dataset = CIFAR100RotationTestDataset(root=root, rotation_range=rotation_range, apply_rotation=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader

def get_cifar100_test_loader(root, batch_size=128, rotation_range=(0, 360)):
    dataset = CIFAR100RotationTestDataset(root=root, rotation_range=rotation_range, apply_rotation=False)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return dataloader


# --- ISIC 2019 ---

# Using ImageNet stats as a common practice for ISIC
ISIC_MEAN = [0.485, 0.456, 0.406]
ISIC_STD  = [0.229, 0.224, 0.225]

class ISICBase(Dataset):
    def __init__(self, root, train, metadata_filename, image_foldername):
        self.root = root
        self.train = train
        self.image_dir = os.path.join(root, image_foldername)
        self.metadata_path = os.path.join(root, metadata_filename)

        self.classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        try:
            metadata_df = pd.read_csv(self.metadata_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata CSV not found at: {self.metadata_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading metadata CSV {self.metadata_path}: {e}")

        if 'image' not in metadata_df.columns:
            raise ValueError(f"Metadata CSV '{self.metadata_path}' missing required 'image' column.")

        class_columns_present = [col for col in self.classes if col in metadata_df.columns]
        missing_class_cols = [col for col in self.classes[:-1] if col not in metadata_df.columns]

        if missing_class_cols and self.train:
             print(f"Warning: Training metadata CSV '{self.metadata_path}' missing class columns: {missing_class_cols}.")
        elif not class_columns_present and not self.train:
             print(f"Info: Test metadata CSV '{self.metadata_path}' has no class columns. Labels will be assigned as 'UNK'.")


        self.image_files = []
        self.labels = []

        print(f"Loading ISIC metadata from: {self.metadata_path}")
        loaded_count = 0
        missing_file_count = 0
        missing_label_count = 0

        for index, row in metadata_df.iterrows():
            img_name_base = row['image']
            img_name = img_name_base + ".jpg"
            img_path = os.path.join(self.image_dir, img_name)

            if os.path.exists(img_path):
                self.image_files.append(img_path)
                label_idx = -1
                if class_columns_present:
                    for cls_name in class_columns_present:
                        if cls_name in row and pd.notna(row[cls_name]) and row[cls_name] == 1.0:
                            label_idx = self.class_to_idx[cls_name]
                            break

                if label_idx == -1:
                    label_idx = self.class_to_idx['UNK']
                    if class_columns_present:
                         missing_label_count += 1

                self.labels.append(label_idx)
                loaded_count += 1
            else:
                missing_file_count += 1

        print(f"Found {loaded_count} images corresponding to metadata entries.")
        if missing_file_count > 0:
            print(f"Warning: Skipped {missing_file_count} entries due to missing image files in '{self.image_dir}'.")
        if missing_label_count > 0:
            print(f"Warning: Assigned 'UNK' label to {missing_label_count} images as no specific class label (1.0) was found in metadata rows.")
        if not self.image_files:
             raise ValueError(f"No images found. Check image directory '{self.image_dir}' and metadata '{self.metadata_path}'.")


    def __len__(self):
        return len(self.image_files)

    def _load_image(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

    def _preprocess(self, pil_img):
        img = TF.resize(pil_img, (224, 224))
        return TF.to_tensor(img)

    def _normalize(self, tensor_img):
        return TF.normalize(tensor_img, ISIC_MEAN, ISIC_STD)

class ISICTrainDataset(ISICBase):
    def __init__(self, root, metadata_filename='ISIC_2019_Training_GroundTruth.csv', image_foldername='ISIC_2019_Training_Input'):
        super().__init__(root, train=True, metadata_filename=metadata_filename, image_foldername=image_foldername)
        self.augmentations = T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

    def __getitem__(self, idx):
        pil_img = self._load_image(idx)
        if pil_img is None:
            print(f"Warning: Could not load image at index {idx}. Returning item 0 instead.")
            idx = 0
            pil_img = self._load_image(idx)
            if pil_img is None:
                 raise RuntimeError("Failed to load even the first image. Check dataset integrity.")

        label = self.labels[idx]
        img = self._preprocess(pil_img)
        img = self.augmentations(img)
        img = self._normalize(img)
        return img, label

class ISICTestDataset(ISICBase):
    def __init__(self, root, metadata_filename='ISIC_2019_Test_GroundTruth.csv', image_foldername='ISIC_2019_Test_Input'):
        super().__init__(root, train=False, metadata_filename=metadata_filename, image_foldername=image_foldername)

    def __getitem__(self, idx):
        pil_img = self._load_image(idx)
        if pil_img is None:
            print(f"Warning: Could not load image at index {idx}. Returning item 0 instead.")
            idx = 0
            pil_img = self._load_image(idx)
            if pil_img is None:
                 raise RuntimeError("Failed to load even the first image. Check dataset integrity.")

        label = self.labels[idx]
        img = self._preprocess(pil_img)

        angle = 0.0
        img = self._normalize(img)

        return img, label, angle

def get_isic_train_loader(root, metadata_filename='ISIC_2019_Training_GroundTruth.csv', image_foldername='ISIC_2019_Training_Input', batch_size=128, num_workers=4):
    dataset = ISICTrainDataset(root=root, metadata_filename=metadata_filename, image_foldername=image_foldername)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader

def get_isic_test_loader(root, metadata_filename='ISIC_2019_Test_GroundTruth.csv', image_foldername='ISIC_2019_Test_Input', batch_size=128, num_workers=4):
    dataset = ISICTestDataset(
        root=root,
        metadata_filename=metadata_filename,
        image_foldername=image_foldername,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


# --- TEM 2019 ---

TEM_MEAN = [0.485, 0.456, 0.406]
TEM_STD  = [0.229, 0.224, 0.225]

class TEM2019Base(Dataset):
    def __init__(self, root, dataset_name="TEM_virus_dataset", split='train'):
        self.root = root
        self.split = split
        self.dataset_name = dataset_name
        self.image_dir = os.path.join(root, dataset_name, split)

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Dataset directory not found at: {self.image_dir}")
        
        self.classes = []
        for item in os.listdir(self.image_dir):
            item_path = os.path.join(self.image_dir, item)
            if os.path.isdir(item_path):
                self.classes.append(item)
        
        self.classes.sort()
        
        if not self.classes:
            raise ValueError(f"No class subdirectories found in {self.image_dir}")
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_files = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.image_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.tif', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_files.append(img_path)
                    self.labels.append(class_idx)
                    
        print(f"Loaded {len(self.image_files)} images from {len(self.classes)} classes in TEM dataset ({split} split)")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.image_files)

    def _load_image(self, idx):
        img_path = self.image_files[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

    def _preprocess(self, pil_img):
        img = TF.resize(pil_img, (224, 224))
        return TF.to_tensor(img)

    def _normalize(self, tensor_img):
        return TF.normalize(tensor_img, TEM_MEAN, TEM_STD)
    
    def __getitem__(self, idx):
        img = self._load_image(idx)
        if img is None:
            print(f"Warning: Could not load image at index {idx}. Returning item 0 instead.")
            idx = 0
            img = self._load_image(idx)
            if img is None:
                raise RuntimeError("Failed to load even the first image. Check dataset integrity.")
        
        img = self._preprocess(img)
        img = self._normalize(img)
        
        return img, self.labels[idx]

class TEMTrainDataset(TEM2019Base):
    def __init__(self, root, **kwargs):
        super().__init__(root=root, split='train', **kwargs)
        self.augmentations = T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])
        
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        img = self.augmentations(img)
            
        return img, label

class TEMTestDataset(TEM2019Base):
    def __init__(self, root, **kwargs):
        super().__init__(root=root, split='test', **kwargs)
        
    def __getitem__(self, idx):
        return super().__getitem__(idx)


def get_tem_train_loader(root, batch_size=32):
    dataset = TEMTrainDataset(root=root)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader


def get_tem_test_loader(root, batch_size=32):
    dataset = TEMTestDataset(root=root)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader



if __name__ == "__main__":
    print("--- Testing ISIC Dataset Loading - Showing First Item ---")
    isic_root_dir = './data/isic'
    num_samples_to_show = 1

    try:
        isic_train_dataset = ISICTrainDataset(
            root=isic_root_dir,
            metadata_filename='ISIC_2019_Training_GroundTruth.csv',
            image_foldername='ISIC_2019_Training_Input'
        )

        if len(isic_train_dataset) == 0:
             print("Dataset is empty, cannot display samples.")
        else:
            print(f"Loading the first sample (index 0)...")
            img_tensor, label_idx = isic_train_dataset[0]

            img_path = isic_train_dataset.image_files[0]
            img_filename_base = os.path.basename(img_path).split('.')[0]
            label_name = isic_train_dataset.classes[label_idx]

            print("\n--- Item Details ---")
            print(f"Image Tensor Shape: {img_tensor.shape}")
            print(f"Image Tensor Dtype: {img_tensor.dtype}")
            print(f"Image Tensor Min:   {img_tensor.min():.4f}")
            print(f"Image Tensor Max:   {img_tensor.max():.4f}")
            print(f"Image Tensor Mean:  {img_tensor.mean():.4f}")
            print(f"Label Index:        {label_idx}")
            print(f"Label Name:         '{label_name}'")
            print(f"Original Filename:  '{img_filename_base}.jpg'")
            print("--------------------\n")

            mean = torch.tensor(ISIC_MEAN).view(3, 1, 1)
            std = torch.tensor(ISIC_STD).view(3, 1, 1)
            denormalized_image = img_tensor * std + mean
            denormalized_image = torch.clamp(denormalized_image, 0, 1)

            np_img = denormalized_image.numpy()
            plt.figure(figsize=(6, 6))
            plt.imshow(np.transpose(np_img, (1, 2, 0)))
            plt.title(f"Sample 0: '{img_filename_base}' - Label: '{label_name}' ({label_idx})\n(Augmented & Normalized -> Denormalized)")
            plt.axis('off')
            save_filename = f"{img_filename_base}_label_{label_name}_augmented_denormalized.png"
            save_path = os.path.join(isic_root_dir, save_filename)
            print(f"Saving the (denormalized) image to: {save_path}")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


    except FileNotFoundError as e:
        print(f"\nError: Could not find ISIC data.")
        print(f"Please ensure the ISIC data is located at: '{isic_root_dir}'")
        print(f"Expected structure:")
        print(f"  {isic_root_dir}/")
        print(f"  ├── ISIC_2019_Training_Input/     (or your specified image_foldername)")
        print(f"  │   └── *.jpg")
        print(f"  └── ISIC_2019_Training_GroundTruth.csv (or your specified metadata_filename)")
        print(f"Original error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during ISIC testing: {e}")

