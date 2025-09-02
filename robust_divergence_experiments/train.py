import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import copy

# Adjusting path to import from parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.autoencoder import Autoencoder
import endoscopycorruptions.corruptions as semantic_corruptions
import config

# --- Full list of corruptions for evaluation ---
CORRUPTION_FUNCTIONS = {
    'brightness': semantic_corruptions.brightness,
    'darkness': semantic_corruptions.darkness,
    'contrast': semantic_corruptions.contrast,
    'fog': semantic_corruptions.fog,

    'defocus_blur': semantic_corruptions.defocus_blur,
    'glass_blur': semantic_corruptions.glass_blur,
    'motion_blur': semantic_corruptions.motion_blur,
    'zoom_blur': semantic_corruptions.zoom_blur,


    'gaussian_noise': semantic_corruptions.gaussian_noise,
    'impulse_noise': semantic_corruptions.impulse_noise,
    'shot_noise': semantic_corruptions.shot_noise,
    'iso_noise': semantic_corruptions.iso_noise,

    'lens_distortion': semantic_corruptions.lens_distortion,
    'resolution_change': semantic_corruptions.resolution_change,
    'specular_reflection': semantic_corruptions.specular_reflection,
    'color_changes': semantic_corruptions.color_changes,

    # 'speckle_noise': semantic_corruptions.speckle_noise,
    # 'gaussian_blur': semantic_corruptions.gaussian_blur,
    # 'elastic_transform': semantic_corruptions.elastic_transform,
    # 'pixelate': semantic_corruptions.pixelate,
    # 'jpeg_compression': semantic_corruptions.jpeg_compression,
    # 'spatter': semantic_corruptions.spatter,
}

# --- Re-used Utility Functions ---

def get_data_paths(dataset_name, subtype, data_split='train'):
    # ... (code from previous script)
    paths, labels = [], []
    dataset_path = os.path.join(config.DATA_ROOT, dataset_name, subtype, data_split)
    if not os.path.exists(dataset_path): return paths, labels
    class_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(class_path, img_file))
                labels.append(class_folder)
    return paths, labels

class KidneyStoneDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, corruption_info=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.corruption_info = corruption_info

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.corruption_info:
            corruption_func, severity = self.corruption_info
            image_np = np.array(image)
            corrupted_np = corruption_func(image_np, severity)
            image = Image.fromarray(corrupted_np.astype('uint8'), 'RGB')

        if self.transform:
            image = self.transform(image)
        return image, label

def apply_corruption_on_tensor(tensor_batch, corruption_func, severity):
    # ... (code from previous script)
    corrupted_batch = []
    for i in range(tensor_batch.size(0)):
        tensor_image = tensor_batch[i]
        pil_image = transforms.ToPILImage()(tensor_image.cpu())
        np_image = np.array(pil_image)
        corrupted_np = corruption_func(np_image, severity)
        corrupted_pil = Image.fromarray(corrupted_np.astype('uint8'), 'RGB')
        corrupted_tensor = transforms.ToTensor()(corrupted_pil)
        corrupted_batch.append(corrupted_tensor)
    return torch.stack(corrupted_batch)

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    all_results = []
    
    for corruption_name, corruption_func in CORRUPTION_FUNCTIONS.items():
        for severity in range(1, 6):
            all_errors_clean, all_errors_corrupted = [], []
            with torch.no_grad():
                for images, _ in tqdm(test_loader, desc=f"Evaluating {corruption_name} (sev {severity})"):
                    images = images.to(device)
                    corrupted_images = apply_corruption_on_tensor(images, corruption_func, severity).to(device)
                    
                    error_clean = criterion(model(images)[0], images).mean(dim=[1,2,3])
                    error_corrupted = criterion(model(corrupted_images)[0], corrupted_images).mean(dim=[1,2,3])
                    
                    all_errors_clean.extend(error_clean.cpu().numpy())
                    all_errors_corrupted.extend(error_corrupted.cpu().numpy())
            
            y_true = np.concatenate([np.zeros(len(all_errors_clean)), np.ones(len(all_errors_corrupted))])
            y_scores = np.concatenate([all_errors_clean, all_errors_corrupted])
            
            # --- New Metrics Calculation ---
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            
            # Find optimal threshold from ROC curve (Youden's J statistic)
            if len(thresholds) > 1:
                j_scores = tpr - fpr
                best_idx = np.argmax(j_scores)
                best_threshold = thresholds[best_idx]
            else:
                best_threshold = 0.5 # Fallback if only one class is present
            
            # Get predictions based on the optimal threshold
            y_pred = (y_scores >= best_threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            auc = roc_auc_score(y_true, y_scores)

            all_results.append({
                'corruption': corruption_name, 
                'severity': severity, 
                'auc': auc,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'threshold': best_threshold
            })
            print(f"  {corruption_name} (sev {severity}) - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    return all_results

# --- New Federated Learning Logic ---

def create_client_dataloaders(dataset_name, subtype, num_clients, clients_with_corruption):
    transform = transforms.Compose([transforms.Resize(config.IMAGE_SIZE), transforms.ToTensor()])
    
    train_paths, train_labels = get_data_paths(dataset_name, subtype, 'train')
    
    # Create a base dataset
    base_dataset = KidneyStoneDataset(train_paths, train_labels, transform)
    
    # Split indices for clients
    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    client_indices = np.array_split(indices, num_clients)
    
    client_loaders = []
    # Get a list of corruption functions to choose from
    corruption_function_list = list(CORRUPTION_FUNCTIONS.values())

    for i in range(num_clients):
        corruption_info = None
        if i in clients_with_corruption:
            # Randomly select a corruption for this client
            corruption_func = random.choice(corruption_function_list)
            severity = random.randint(3, 5) # Persistent, strong corruption
            corruption_info = (corruption_func, severity)
            print(f"Client {i} will have persistent '{corruption_func.__name__}' corruption with severity {severity}.")
        
        client_dataset = KidneyStoneDataset(
            [train_paths[j] for j in client_indices[i]],
            [train_labels[j] for j in client_indices[i]],
            transform,
            corruption_info
        )
        loader = DataLoader(client_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        client_loaders.append(loader)
        
    return client_loaders

def client_update(client_loader, model, device):
    """Simulates a single client's training process."""
    local_model = copy.deepcopy(model)
    local_model.train()
    optimizer = torch.optim.Adam(local_model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Initial Benchmark Error Calculation
    initial_error = 0
    with torch.no_grad():
        for images, _ in client_loader:
            images = images.to(device)
            reconstructed, _ = model(images)
            initial_error += criterion(reconstructed, images).item()
    initial_benchmark_error = initial_error / len(client_loader)
    
    # Local Training
    for _ in range(config.LOCAL_EPOCHS):
        for images, _ in client_loader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructed, _ = local_model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            
    return local_model.state_dict(), initial_benchmark_error

def get_model_divergence(global_weights, client_weights):
    """Calculates the cosine distance between two state dicts."""
    global_vec = nn.utils.parameters_to_vector(global_weights.values())
    client_vec = nn.utils.parameters_to_vector(client_weights.values())
    return 1 - F.cosine_similarity(global_vec.unsqueeze(0), client_vec.unsqueeze(0)).item()

def main():
    output_dir = "robust_divergence_experiments/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through each dataset and subversion to train 6 distinct models
    for dataset_name in config.DATASETS:
        for subtype in config.SUBVERSIONS:
            model_name = f"FED_{dataset_name.split(' ')[0]}_{subtype}_Divergence"
            print(f"--- Training Federated Model: {model_name} ---")

            # --- Setup Federated Environment ---
            NUM_CLIENTS = config.NUM_CLIENTS
            NUM_CORRUPTED_CLIENTS = int(0.3 * NUM_CLIENTS) # 30% of clients have bad data
            all_client_indices = list(range(NUM_CLIENTS))
            # Randomly select which clients will have corrupted data
            clients_with_corruption = random.sample(all_client_indices, NUM_CORRUPTED_CLIENTS)
            print(f"Randomly selected {len(clients_with_corruption)} clients to have corrupted data: {clients_with_corruption}")

            client_loaders = create_client_dataloaders(dataset_name, subtype, NUM_CLIENTS, clients_with_corruption)
            
            # --- Global Model and Test Loader ---
            global_model = Autoencoder(input_channels=config.CHANNELS, latent_dim=config.LATENT_DIM).to(config.DEVICE)
            transform = transforms.Compose([transforms.Resize(config.IMAGE_SIZE), transforms.ToTensor()])
            test_paths, test_labels = get_data_paths(dataset_name, subtype, 'test')
            test_dataset = KidneyStoneDataset(test_paths, test_labels, transform=transform)
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

            # --- Federated Training Loop ---
            for round_num in range(config.NUM_ROUNDS):
                print(f"\n--- Round {round_num + 1}/{config.NUM_ROUNDS} ---")
                
                # Select a random subset of clients for this round
                participating_clients = random.sample(all_client_indices, config.CLIENTS_PER_ROUND)
                print(f"  Selected {len(participating_clients)} clients for this round: {participating_clients}")

                global_weights = copy.deepcopy(global_model.state_dict())
                client_updates = []
                
                for client_idx in participating_clients:
                    print(f"  Training client {client_idx}...")
                    client_weights, benchmark_error = client_update(client_loaders[client_idx], global_model, config.DEVICE)
                    divergence = get_model_divergence(global_weights, client_weights)
                    client_updates.append({
                        'weights': client_weights,
                        'benchmark_error': benchmark_error,
                        'divergence': divergence
                    })
                    print(f"    -> Client {client_idx}: Benchmark Error={benchmark_error:.4f}, Divergence={divergence:.4f}")
                    
                # --- Server-Side Aggregation ---
                total_weight = 0
                new_global_weights = {k: torch.zeros_like(v) for k, v in global_weights.items()}
                
                for update in client_updates:
                    # Trust score is inversely proportional to benchmark error and divergence
                    trust_score = 1 / ((update['benchmark_error'] + 1e-6) * (update['divergence'] + 1e-6))
                    total_weight += trust_score
                    
                    for k in new_global_weights.keys():
                        if torch.is_floating_point(new_global_weights[k]):
                            new_global_weights[k] += update['weights'][k] * trust_score
                
                # Get the latest global model state dict to preserve non-float buffers
                final_weights = global_model.state_dict()
                if total_weight > 0:
                    for k in final_weights.keys():
                        if torch.is_floating_point(final_weights[k]):
                            final_weights[k] = new_global_weights[k] / total_weight
                    
                global_model.load_state_dict(final_weights)
                print("Global model updated.")
                
            # --- Final Evaluation ---
            print(f"--- Final Model Evaluation for {model_name} ---")
            evaluation_results = evaluate_model(global_model, test_loader, config.DEVICE)
            
            df = pd.DataFrame(evaluation_results)
            df.to_csv(os.path.join(output_dir, f"{model_name}_results.csv"), index=False)
            print(f"Results saved to {os.path.join(output_dir, f'{model_name}_results.csv')}")
            
            # Save final model
            model_save_path = os.path.join(output_dir, f"{model_name}.pth")
            torch.save(global_model.state_dict(), model_save_path)
            print(f"Final model saved to {model_save_path}")

if __name__ == '__main__':
    main() 