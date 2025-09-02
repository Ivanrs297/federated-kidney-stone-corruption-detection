"""
Simulation script for federated autoencoder training
This script runs the entire federated learning process in simulation mode
"""

import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import only specific Flower components to avoid TensorFlow
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes, EvaluateRes

from client import AutoencoderClient
from server import AutoencoderStrategy
from utils import create_client_dataloaders, create_global_test_loader
import config


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For reproducibility, set deterministic mode
    # Note: This may reduce performance but ensures reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_federated_training(clients, strategy, num_rounds):
    """Run federated training without Ray simulation"""
    
    print(f"Starting federated training for {num_rounds} rounds...")
    
    # Initialize global parameters
    global_parameters = strategy.initialize_parameters(None)
    
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")
        
        # Select clients for this round
        num_clients_this_round = min(config.CLIENTS_PER_ROUND, len(clients))
        selected_clients = random.sample(clients, num_clients_this_round)
        
        # Training phase
        fit_results = []
        for client in selected_clients:
            print(f"\n📚 Training Client {client.client_id} (Round {round_num})...")
            
            # Convert parameters for client
            parameters_list = parameters_to_ndarrays(global_parameters)
            
            # Train client
            updated_parameters, num_examples, metrics = client.fit(parameters_list, {})
            
            # Display training results
            train_loss = metrics.get('train_loss', 'N/A')
            print(f"   ✅ Client {client.client_id} training completed - Loss: {train_loss:.6f}, Samples: {num_examples}")
            
            # Create a mock FitRes object that matches Flower's expected format
            fit_res = FitRes(
                status=None,  # Not used in our case
                parameters=ndarrays_to_parameters(updated_parameters),
                num_examples=num_examples,
                metrics=metrics
            )
            
            # Store results in the format expected by Flower strategy
            # We use None as ClientProxy since we're not using distributed clients
            fit_results.append((None, fit_res))
        
        # Aggregate parameters
        global_parameters, aggregated_metrics = strategy.aggregate_fit(round_num, fit_results, [])
        
        # Evaluation phase
        eval_results = []
        for client in selected_clients:
            print(f"\n🔍 Evaluating Client {client.client_id} (Round {round_num})...")
            
            # Convert parameters for client
            parameters_list = parameters_to_ndarrays(global_parameters)
            
            # Evaluate client
            loss, num_examples, metrics = client.evaluate(parameters_list, {})
            
            # Display evaluation results
            test_loss = metrics.get('test_loss', loss)
            print(f"   📊 Client {client.client_id} evaluation - Loss: {test_loss:.6f}, Samples: {num_examples}")
            
            # Create a mock EvaluateRes object that matches Flower's expected format
            eval_res = EvaluateRes(
                status=None,  # Not used in our case
                loss=loss,
                num_examples=num_examples,
                metrics=metrics
            )
            
            # Store results in the format expected by Flower strategy
            # We use None as ClientProxy since we're not using distributed clients
            eval_results.append((None, eval_res))
        
        # Aggregate evaluation results
        strategy.aggregate_evaluate(round_num, eval_results, [])
        
        # GPU memory monitoring
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1e9
            memory_cached = torch.cuda.memory_reserved(0) / 1e9
            print(f"\n💾 GPU Memory - Used: {memory_used:.2f} GB, Cached: {memory_cached:.2f} GB")
            
            # Clear cache periodically to prevent memory buildup
            if round_num % 3 == 0:
                torch.cuda.empty_cache()
        
        print(f"\n✅ Round {round_num}/{num_rounds} completed successfully!")
        print("=" * 60)
    
    return strategy.round_losses


def setup_federated_training(num_clients, corruption_prob, alpha, datasets, subversions):
    """Setup federated training by creating clients and global test loader"""
    print("Setting up federated training...")
    print(f"Number of clients: {num_clients}")
    print(f"Corruption probability: {corruption_prob}")
    print(f"Non-IID alpha: {alpha}")
    print(f"Datasets: {datasets}")
    print(f"Subversions: {subversions}")
    
    # Create client data loaders
    client_loaders = create_client_dataloaders(
        num_clients=num_clients,
        corruption_prob=corruption_prob,
        alpha=alpha,
        datasets=datasets,
        subversions=subversions
    )
    
    # Create client instances
    clients = []
    for client_id, (train_loader, test_loader) in enumerate(client_loaders):
        client = AutoencoderClient(client_id, train_loader, test_loader)
        clients.append(client)
    
    # Create global test loader
    global_test_loader = create_global_test_loader(datasets=datasets, subversions=subversions)
    
    return clients, global_test_loader


def plot_results(losses: List[float], save_path: str = None):
    """Plot training results"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2, marker='o')
    plt.title('Federated Autoencoder Training - Global Test Loss')
    plt.xlabel('Round')
    plt.ylabel('Reconstruction Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def save_results(losses: List[float], config_dict: Dict, save_dir: str = "results"):
    """Save training results and configuration"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save losses
    np.save(os.path.join(save_dir, "training_losses.npy"), losses)
    
    # Save configuration
    with open(os.path.join(save_dir, "config.txt"), "w") as f:
        f.write("Federated Autoencoder Training Configuration\n")
        f.write("=" * 50 + "\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("\nFinal Results:\n")
        f.write(f"Final loss: {losses[-1]:.6f}\n")
        f.write(f"Best loss: {min(losses):.6f}\n")
        f.write(f"Best round: {losses.index(min(losses)) + 1}\n")
    
    print(f"Results saved to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run federated autoencoder simulation (results and plots saved automatically)")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["Michel Daudon (w256 1k v1)", "Jonathan El-Beze (w256 1k v1)", "all"],
                       default=["all"],
                       help="Datasets to use for training (default: all)")
    parser.add_argument("--subversions", nargs="+", 
                       choices=["MIX", "SEC", "SUR", "all"],
                       default=["all"],
                       help="Subversions to use for training (default: all)")
    parser.add_argument("--num_clients", type=int, default=config.NUM_CLIENTS, 
                       help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=config.NUM_ROUNDS, 
                       help="Number of federated rounds")
    parser.add_argument("--corruption_prob", type=float, default=config.CORRUPTION_PROBABILITY, 
                       help="Probability of image corruption")
    parser.add_argument("--alpha", type=float, default=config.ALPHA, 
                       help="Dirichlet distribution parameter for non-IID data")
    parser.add_argument("--clients_per_round", type=int, default=config.CLIENTS_PER_ROUND, 
                       help="Number of clients per round")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(config.SEED)
    
    # Process dataset and subversion arguments
    if "all" in args.datasets:
        selected_datasets = config.DATASETS
    else:
        selected_datasets = args.datasets
    
    if "all" in args.subversions:
        selected_subversions = config.SUBVERSIONS
    else:
        selected_subversions = args.subversions
    
    # Update config with command line arguments
    config.NUM_CLIENTS = args.num_clients
    config.NUM_ROUNDS = args.num_rounds
    config.CLIENTS_PER_ROUND = args.clients_per_round
    
    print("=" * 60)
    print("FEDERATED AUTOENCODER TRAINING")
    print("=" * 60)
    
    # Display device information
    device = torch.device(config.DEVICE)
    print(f"🖥️  Device: {device}")
    if config.CUDA_AVAILABLE:
        print(f"🚀 GPU: {config.CUDA_DEVICE_NAME}")
        print(f"💾 GPU Memory: {config.CUDA_MEMORY_GB:.1f} GB")
        # Clear GPU cache before starting
        torch.cuda.empty_cache()
    else:
        print("⚠️  CUDA not available, using CPU")
    print()
    
    # Setup federated training
    clients, global_test_loader = setup_federated_training(
        args.num_clients, 
        args.corruption_prob, 
        args.alpha,
        selected_datasets,
        selected_subversions
    )
    
    # Create strategy
    strategy = AutoencoderStrategy(global_test_loader)
    
    print(f"\nStarting federated training with {args.num_clients} clients for {args.num_rounds} rounds...")
    
    # Run federated training (no Ray/simulation)
    losses = run_federated_training(
        clients=clients,
        strategy=strategy,
        num_rounds=args.num_rounds
    )
    
    # Get training summary (no additional model saving)
    summary = strategy.get_training_summary()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Final global test loss: {summary['final_loss']:.6f}")
    print(f"Best global test loss: {summary['best_loss']:.6f} (Round {summary['best_round']})")
    print(f"🏆 Only the BEST model has been saved to: models/best_autoencoder.pth")
    
    # Save configuration
    config_dict = {
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "clients_per_round": args.clients_per_round,
        "corruption_probability": args.corruption_prob,
        "alpha": args.alpha,
        "latent_dim": config.LATENT_DIM,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "local_epochs": config.LOCAL_EPOCHS,
        "image_size": config.IMAGE_SIZE,
    }
    
    # Always save results and plot results
    print("\n📊 Saving results and generating plots...")
    save_results(losses, config_dict)
    plot_path = "results/training_plot.png"
    plot_results(losses, plot_path)
    
    return losses


if __name__ == "__main__":
    main() 