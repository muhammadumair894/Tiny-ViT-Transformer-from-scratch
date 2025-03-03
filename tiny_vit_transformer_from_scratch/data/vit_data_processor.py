import os
import torch

from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class DataPreprocessor:
    """
    Loads ImageNet 1K training and validation sets from the standard folder structure.
    """
    def __init__(self, config):
        """
        :param config: A config object with fields:
            - data_dir: Path to the root of ImageNet (containing train/ and val/).
            - batch_size
            - num_workers
            - pin_memory
            - im_size
        """
        self.data_dir = config.data_dir           # e.g. "/path/to/imagenet"
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory
        self.im_size = config.im_size

        # Standard ImageNet train augmentation
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(
                self.im_size, 
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # Standard ImageNet validation/test augmentation
        self.val_transforms = transforms.Compose([
            # Often we do a slightly larger resize, e.g. 256, and then center-crop
            transforms.Resize(
                int(self.im_size * 256/224),  # e.g. 256 when im_size=224
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(self.im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def create_dataloaders(self):
        """
        Creates train and validation data loaders from the official ImageNet folders.
        (If you need a separate test set, you can do so similarly.)
        """
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")

        # Use ImageFolder if your directories are structured as:
        #   imagenet/train/class_x/*.jpg
        #   imagenet/val/class_x/*.jpg
        train_dataset = ImageFolder(root=train_dir, transform=self.train_transforms)
        val_dataset = ImageFolder(root=val_dir,   transform=self.val_transforms)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,           # Shuffle only for training
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        return train_loader, val_loader

    def get_class_names(self):
        """
        Optional helper to retrieve class names from the folder structure
        (only works after you've created the dataset).
        """
        train_dir = os.path.join(self.data_dir, "train")
        dataset = ImageFolder(root=train_dir)
        return dataset.classes
