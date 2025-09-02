"""
Flower server for federated autoencoder training
"""

import torch
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict
from flwr.common import Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from models import Autoencoder
from utils.metrics import calculate_comprehensive_metrics, print_metrics_summary
import config


class AutoencoderStrategy(FedAvg):
    """Custom strategy for autoencoder federated learning"""
    
    def __init__(self, global_test_loader=None, experiment_name="", datasets=None):
        super().__init__(
            fraction_fit=config.CLIENTS_PER_ROUND / config.NUM_CLIENTS,
            fraction_evaluate=1.0,
            min_fit_clients=config.CLIENTS_PER_ROUND,
            min_evaluate_clients=config.CLIENTS_PER_ROUND,
            min_available_clients=config.NUM_CLIENTS,
        )
        
        self.global_test_loader = global_test_loader
        self.experiment_name = experiment_name
        self.datasets = datasets or []
        
        # Initialize global model for evaluation
        self.global_model = Autoencoder(
            input_channels=config.CHANNELS,
            latent_dim=config.LATENT_DIM
        )
        
        self.device = torch.device(config.DEVICE)
        self.global_model.to(self.device)
        
        print(f"Global model initialized for {experiment_name} on {self.device}")
        
        # Loss function for evaluation
        self.criterion = torch.nn.MSELoss()
        
        # Track metrics and best model
        self.round_losses = []
        self.best_loss = float('inf')
        self.best_round = 0
        self.best_model_state = None
        self.best_metrics = None
        
        # Create model save directory for this experiment
        self.model_save_dir = os.path.join(config.MODEL_SAVE_DIR, experiment_name)
        os.makedirs(self.model_save_dir, exist_ok=True)
        
    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters"""
        parameters = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
        return ndarrays_to_parameters(parameters)
    
    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate fit results using weighted average"""
        
        if not results:
            return None, {}
        
        # Call parent aggregate_fit
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            # Update global model with aggregated parameters
            parameters_list = parameters_to_ndarrays(aggregated_parameters)
            params_dict = zip(self.global_model.state_dict().keys(), parameters_list)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.global_model.load_state_dict(state_dict, strict=True)
            
            # Evaluate global model
            if self.global_test_loader is not None:
                global_loss = self.evaluate_global_model()
                self.round_losses.append(global_loss)
                
                print(f"Round {server_round} ({self.experiment_name}): Global test loss = {global_loss:.6f}")
                
                # Check if this is the best model so far - ONLY save if it's the best
                if global_loss < self.best_loss:
                    self.best_loss = global_loss
                    self.best_round = server_round
                    self.best_model_state = self.global_model.state_dict().copy()
                    
                    # Calculate comprehensive metrics for the best model
                    self.best_metrics = calculate_comprehensive_metrics(
                        self.global_model, self.global_test_loader, self.device
                    )
                    
                    # Save ONLY the best model with comprehensive metrics
                    self.save_best_model()
                    print(f"🏆 New best model for {self.experiment_name}! Loss: {global_loss:.6f} (Round {server_round})")
                    
                    # Print metrics summary
                    print_metrics_summary(self.best_metrics, self.experiment_name)
                else:
                    print(f"   Current best for {self.experiment_name}: {self.best_loss:.6f} (Round {self.best_round})")
                
                # Add global metrics
                aggregated_metrics["global_test_loss"] = global_loss
                aggregated_metrics["best_loss"] = self.best_loss
                aggregated_metrics["best_round"] = self.best_round
        
        return aggregated_parameters, aggregated_metrics
    
    def save_best_model(self):
        """Save the best performing model with comprehensive metrics"""
        if self.best_model_state is not None and self.best_metrics is not None:
            best_model_path = os.path.join(self.model_save_dir, f"best_autoencoder_{self.experiment_name}.pth")
            
            # Save model state dict and comprehensive metadata
            torch.save({
                'model_state_dict': self.best_model_state,
                'experiment_name': self.experiment_name,
                'datasets': self.datasets,
                'best_loss': self.best_loss,
                'best_round': self.best_round,
                'metrics': self.best_metrics,  # Comprehensive metrics
                'model_config': {
                    'input_channels': config.CHANNELS,
                    'latent_dim': config.LATENT_DIM,
                    'image_size': config.IMAGE_SIZE
                },
                'training_config': {
                    'num_clients': config.NUM_CLIENTS,
                    'num_rounds': config.NUM_ROUNDS,
                    'clients_per_round': config.CLIENTS_PER_ROUND,
                    'local_epochs': config.LOCAL_EPOCHS,
                    'learning_rate': config.LEARNING_RATE,
                    'batch_size': config.BATCH_SIZE,
                    'corruption_probability': config.CORRUPTION_PROBABILITY,
                    'alpha': config.ALPHA
                }
            }, best_model_path)
            
            print(f"💾 Best model for {self.experiment_name} saved to {best_model_path}")
    
    def get_training_summary(self):
        """Get a summary of the training process"""
        return {
            'experiment_name': self.experiment_name,
            'best_loss': self.best_loss,
            'best_round': self.best_round,
            'best_metrics': self.best_metrics,
            'total_rounds': len(self.round_losses),
            'round_losses': self.round_losses,
            'final_loss': self.round_losses[-1] if self.round_losses else None
        }
    
    def evaluate_global_model(self) -> float:
        """Evaluate the global model on the test set"""
        
        self.global_model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for data, _ in self.global_test_loader:
                data = data.to(self.device)
                
                # Forward pass
                reconstructed, latent = self.global_model(data)
                
                # Calculate loss
                loss = self.criterion(reconstructed, data)
                
                total_loss += loss.item() * data.size(0)
                num_samples += data.size(0)
        
        return total_loss / num_samples if num_samples > 0 else 0.0
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation results"""
        
        if not results:
            return None, {}
        
        # Calculate weighted average of client losses
        total_loss = 0.0
        total_samples = 0
        
        for _, evaluate_res in results:
            # evaluate_res is now an EvaluateRes object, not a tuple
            loss = evaluate_res.loss
            num_samples = evaluate_res.num_examples
            total_loss += loss * num_samples
            total_samples += num_samples
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        print(f"Round {server_round} ({self.experiment_name}): Average client test loss = {avg_loss:.6f}")
        
        return avg_loss, {"client_test_loss": avg_loss}


# Note: start_server function removed since we're using direct federated training
# without distributed server/client architecture


if __name__ == "__main__":
    print("Note: This server.py file now only contains the AutoencoderStrategy class.")
    print("Use run_simulation.py for federated training without distributed server/client setup.") 