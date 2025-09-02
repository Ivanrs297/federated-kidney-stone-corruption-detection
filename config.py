"""
Configuration file for Federated Autoencoder Training
"""

import os

# Dataset Configuration
DATA_ROOT = "data"
DATASETS = ["Michel Daudon (w256 1k v1)", "Jonathan El-Beze (w256 1k v1)"]
SUBVERSIONS = ["MIX", "SEC", "SUR"]
IMAGE_SIZE = (256, 256)
CHANNELS = 3

# Federated Learning Configuration
NUM_CLIENTS = 20  # Default number of clients
NUM_ROUNDS = 20  # Number of federated rounds
CLIENTS_PER_ROUND = 10  # Number of clients participating per round
CORRUPTED_CLIENT_RATIO = 0.3  # Ratio of clients with persistent data corruption

# Model Configuration
LATENT_DIM = 128  # Autoencoder latent dimension
LEARNING_RATE = 0.001
BATCH_SIZE = 32
LOCAL_EPOCHS = 3  # Number of local epochs per client

# Data Corruption Configuration
CORRUPTION_PROBABILITY = 0.2  # Default corruption probability
CORRUPTION_TYPES = [
    "gaussian_noise",
    "salt_pepper",
    "blur",
    "brightness",
    "contrast"
]

# Training Configuration
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# CUDA Configuration
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    CUDA_DEVICE_COUNT = torch.cuda.device_count()
    CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
    CUDA_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    
    # Set memory allocation strategy
    torch.cuda.empty_cache()  # Clear cache

# Logging Configuration
LOG_DIR = "logs"
MODEL_SAVE_DIR = "models"
RESULTS_DIR = "results"

# Non-IID Configuration
ALPHA = 0.3  # Dirichlet distribution parameter for non-IID data distribution
MIN_SAMPLES_PER_CLIENT = 50  # Minimum samples per client 