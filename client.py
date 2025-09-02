"""
Flower client for federated autoencoder training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from flwr.client import NumPyClient
from models import Autoencoder
import config


class AutoencoderClient(NumPyClient):
    """Flower client for autoencoder training"""
    
    def __init__(self, client_id, train_loader, test_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Initialize model
        self.model = Autoencoder(
            input_channels=config.CHANNELS,
            latent_dim=config.LATENT_DIM
        )
        
        # Move to device
        self.device = torch.device(config.DEVICE)
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        
        print(f"   🤖 Client {client_id} initialized with {len(train_loader.dataset)} training samples on {self.device}")
    
    def get_parameters(self, config_dict=None):
        """Return model parameters as a list of NumPy ndarrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy ndarrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config_dict):
        """Train the model on the locally held training set"""
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Set model to training mode
        self.model.train()
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(config.LOCAL_EPOCHS):
            epoch_loss = 0.0
            
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                reconstructed, latent = self.model(data)
                
                # Calculate loss (reconstruction loss)
                loss = self.criterion(reconstructed, data)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
            
            print(f"      📈 Epoch {epoch + 1}/{config.LOCAL_EPOCHS}, Loss: {epoch_loss / len(self.train_loader):.6f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return (
            self.get_parameters(),
            len(self.train_loader.dataset),
            {"train_loss": avg_loss}
        )
    
    def evaluate(self, parameters, config_dict):
        """Evaluate the model on the locally held test set"""
        
        # Set model parameters
        self.set_parameters(parameters)
        
        # Set model to evaluation mode
        self.model.eval()
        
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                
                # Forward pass
                reconstructed, latent = self.model(data)
                
                # Calculate loss
                loss = self.criterion(reconstructed, data)
                
                total_loss += loss.item() * data.size(0)
                num_samples += data.size(0)
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        # Removed individual client evaluation print as it's now handled in run_simulation.py
        
        return avg_loss, num_samples, {"test_loss": avg_loss}


def create_client(client_id, train_loader, test_loader):
    """Create a Flower client"""
    return AutoencoderClient(client_id, train_loader, test_loader) 