"""
Data utilities for federated autoencoder training
"""

import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from collections import Counter
import config


class ImageCorruption:
    """Class to handle various image corruptions"""
    
    def __init__(self, corruption_prob=0.1):
        self.corruption_prob = corruption_prob
        
    def gaussian_noise(self, image):
        """Add Gaussian noise to image"""
        if random.random() < self.corruption_prob:
            noise = torch.randn_like(image) * 0.1
            image = torch.clamp(image + noise, 0, 1)
        return image
    
    def salt_pepper_noise(self, image):
        """Add salt and pepper noise"""
        if random.random() < self.corruption_prob:
            noise = torch.rand_like(image)
            salt = noise > 0.95
            pepper = noise < 0.05
            image[salt] = 1.0
            image[pepper] = 0.0
        return image
    
    def blur(self, image):
        """Apply blur to image"""
        if random.random() < self.corruption_prob:
            # Convert to PIL for blur operation
            if isinstance(image, torch.Tensor):
                image_pil = transforms.ToPILImage()(image)
                image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))
                image = transforms.ToTensor()(image_pil)
        return image
    
    def brightness_change(self, image):
        """Change brightness of image"""
        if random.random() < self.corruption_prob:
            factor = random.uniform(0.5, 1.5)
            image = torch.clamp(image * factor, 0, 1)
        return image
    
    def contrast_change(self, image):
        """Change contrast of image"""
        if random.random() < self.corruption_prob:
            mean = image.mean()
            factor = random.uniform(0.5, 1.5)
            image = torch.clamp((image - mean) * factor + mean, 0, 1)
        return image
    
    def apply_random_corruption(self, image):
        """Apply a random corruption to the image"""
        corruptions = [
            self.gaussian_noise,
            self.salt_pepper_noise,
            self.blur,
            self.brightness_change,
            self.contrast_change
        ]
        
        corruption_func = random.choice(corruptions)
        return corruption_func(image)


class KidneyStoneDataset(Dataset):
    """Custom dataset for kidney stone images"""
    
    def __init__(self, image_paths, labels, transform=None, corruption_prob=0.0):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.corruption = ImageCorruption(corruption_prob)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Apply corruption if specified
        if self.corruption.corruption_prob > 0:
            image = self.corruption.apply_random_corruption(image)
        
        return image, label


def load_dataset_paths(datasets=None, subversions=None):
    """Load image paths and labels from specified datasets and subversions"""
    all_paths = []
    all_labels = []
    
    # Use all datasets if none specified
    if datasets is None:
        datasets = config.DATASETS
    
    # Use all subversions if none specified
    if subversions is None:
        subversions = config.SUBVERSIONS
    
    for dataset_name in datasets:
        dataset_path = os.path.join(config.DATA_ROOT, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset path does not exist: {dataset_path}")
            continue
        
        for subversion in subversions:
            subversion_path = os.path.join(dataset_path, subversion)
            
            if not os.path.exists(subversion_path):
                print(f"Warning: Subversion path does not exist: {subversion_path}")
                continue
            
            # Load training images (extract class from folder structure)
            train_path = os.path.join(subversion_path, "train")
            if os.path.exists(train_path):
                # Get all class folders in train directory
                class_folders = [d for d in os.listdir(train_path) 
                               if os.path.isdir(os.path.join(train_path, d))]
                
                for class_folder in class_folders:
                    class_path = os.path.join(train_path, class_folder)
                    
                    # Load all images in this class folder
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_file)
                            all_paths.append(img_path)
                            # Create label with class information: "subversion_class"
                            all_labels.append(f"{subversion}_{class_folder}")
            
            # Load test images (extract class from folder structure)
            test_path = os.path.join(subversion_path, "test")
            if os.path.exists(test_path):
                # Get all class folders in test directory
                class_folders = [d for d in os.listdir(test_path) 
                               if os.path.isdir(os.path.join(test_path, d))]
                
                for class_folder in class_folders:
                    class_path = os.path.join(test_path, class_folder)
                    
                    # Load all images in this class folder
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(class_path, img_file)
                            all_paths.append(img_path)
                            # Create label with class information: "subversion_class"
                            all_labels.append(f"{subversion}_{class_folder}")
    
    print(f"📊 Data loading summary:")
    print(f"   Total images: {len(all_paths)}")
    print(f"   Unique classes found: {len(set(all_labels))}")
    print(f"   Classes: {sorted(set(all_labels))}")
    
    return all_paths, all_labels


def redistribute_data_evenly(image_paths, labels, num_clients):
    """Redistribute data evenly among clients as fallback"""
    total_samples = len(image_paths)
    samples_per_client = total_samples // num_clients
    
    # Shuffle data
    combined = list(zip(image_paths, labels))
    np.random.shuffle(combined)
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:  # Last client gets remaining samples
            end_idx = total_samples
        else:
            end_idx = (i + 1) * samples_per_client
        
        client_data = combined[start_idx:end_idx]
        if client_data:
            client_paths, client_labels = zip(*client_data)
            client_datasets.append((list(client_paths), list(client_labels)))
            print(f"Client {i} redistributed with {len(client_paths)} samples")
    
    return client_datasets


def create_non_iid_distribution(image_paths, labels, num_clients, alpha=0.5):
    """Create non-IID data distribution using Dirichlet distribution"""
    
    # Convert labels to numeric
    unique_labels = list(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_to_idx[label] for label in labels]
    
    num_classes = len(unique_labels)
    
    # Create Dirichlet distribution for each client
    client_distributions = np.random.dirichlet([alpha] * num_classes, num_clients)
    
    # Group data by class
    class_indices = {i: [] for i in range(num_classes)}
    for idx, label in enumerate(numeric_labels):
        class_indices[label].append(idx)
    
    # Distribute data to clients
    client_data = [[] for _ in range(num_clients)]
    
    for class_idx in range(num_classes):
        class_data = class_indices[class_idx]
        np.random.shuffle(class_data)
        
        # Calculate how many samples each client gets from this class
        total_samples = len(class_data)
        client_samples = (client_distributions[:, class_idx] * total_samples).astype(int)
        
        # Ensure we don't exceed total samples
        if client_samples.sum() > total_samples:
            excess = client_samples.sum() - total_samples
            client_samples[-1] -= excess
        
        # Distribute samples
        start_idx = 0
        for client_idx, num_samples in enumerate(client_samples):
            if num_samples > 0:
                end_idx = start_idx + num_samples
                client_data[client_idx].extend(class_data[start_idx:end_idx])
                start_idx = end_idx
    
    # Convert indices back to paths and labels
    client_datasets = []
    for client_idx, client_indices in enumerate(client_data):
        if len(client_indices) > 0:  # Accept any client with at least some data
            client_paths = [image_paths[i] for i in client_indices]
            client_labels = [labels[i] for i in client_indices]
            client_datasets.append((client_paths, client_labels))
            print(f"Client {client_idx} will have {len(client_indices)} samples")
        else:
            print(f"Warning: Client {client_idx} has no samples assigned")
    
    # If we don't have enough clients, redistribute the data more evenly
    if len(client_datasets) < num_clients:
        print(f"Warning: Only {len(client_datasets)} clients have sufficient data. Redistributing...")
        return redistribute_data_evenly(image_paths, labels, num_clients)
    
    return client_datasets


def safe_train_test_split(paths, labels, test_size=0.2, random_state=None):
    """
    Safely split data into train/test, handling classes with insufficient samples
    """
    # Count samples per class
    class_counts = Counter(labels)
    
    # Check if we can do stratified split
    min_class_size = min(class_counts.values())
    can_stratify = min_class_size >= 2
    
    if can_stratify:
        try:
            return train_test_split(
                paths, labels, 
                test_size=test_size, 
                random_state=random_state, 
                stratify=labels
            )
        except ValueError as e:
            print(f"   ⚠️ Stratified split failed: {e}")
            can_stratify = False
    
    if not can_stratify:
        print(f"   📊 Using random split (some classes have <2 samples)")
        print(f"   📈 Class distribution: {dict(class_counts)}")
        
        # Use random split without stratification
        return train_test_split(
            paths, labels, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=None
        )


def get_data_transforms():
    """Get data transformations for training and testing"""
    
    train_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def create_client_dataloaders(num_clients, corruption_prob=0.1, alpha=0.5, datasets=None, subversions=None):
    """Create data loaders for all clients with non-IID distribution"""
    
    # Load data from specified datasets and subversions
    all_paths, all_labels = load_dataset_paths(datasets=datasets, subversions=subversions)
    
    print(f"Total images loaded: {len(all_paths)}")
    print(f"Unique labels: {set(all_labels)}")
    
    if len(all_paths) == 0:
        raise ValueError("No images found! Please check your dataset paths and subversions.")
    
    # Create non-IID distribution
    client_datasets = create_non_iid_distribution(all_paths, all_labels, num_clients, alpha)
    
    print(f"Created {len(client_datasets)} client datasets")
    
    # Get transforms
    train_transform, test_transform = get_data_transforms()
    
    # Create data loaders for each client
    client_loaders = []
    
    for i, (client_paths, client_labels) in enumerate(client_datasets):
        print(f"Client {i}: {len(client_paths)} samples")
        
        # Split into train/test for each client using safe splitting
        train_paths, test_paths, train_labels, test_labels = safe_train_test_split(
            client_paths, client_labels, test_size=0.2, random_state=config.SEED
        )
        
        # Create datasets
        train_dataset = KidneyStoneDataset(
            train_paths, train_labels, 
            transform=train_transform, 
            corruption_prob=corruption_prob
        )
        
        test_dataset = KidneyStoneDataset(
            test_paths, test_labels, 
            transform=test_transform, 
            corruption_prob=0.0  # No corruption for test data
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=2
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.BATCH_SIZE, 
            shuffle=False, 
            num_workers=2
        )
        
        client_loaders.append((train_loader, test_loader))
    
    return client_loaders


def create_global_test_loader(datasets=None, subversions=None):
    """Create a global test loader for evaluation"""
    
    # Load data from specified datasets and subversions
    all_paths, all_labels = load_dataset_paths(datasets=datasets, subversions=subversions)
    
    if len(all_paths) == 0:
        raise ValueError("No images found for global test loader! Please check your dataset paths and subversions.")
    
    # Use a subset for global testing with safe splitting
    _, test_paths, _, test_labels = safe_train_test_split(
        all_paths, all_labels, test_size=0.1, random_state=config.SEED
    )
    
    _, test_transform = get_data_transforms()
    
    test_dataset = KidneyStoneDataset(
        test_paths, test_labels, 
        transform=test_transform, 
        corruption_prob=0.0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2
    )
    
    return test_loader 