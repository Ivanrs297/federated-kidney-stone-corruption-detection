"""
Autoencoder model for federated learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder part of the autoencoder"""
    
    def __init__(self, input_channels=3, latent_dim=128):
        super(Encoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1)  # 256 -> 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 128 -> 64
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 64 -> 32
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # 32 -> 16
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)  # 16 -> 8
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(1024)
        
        # Final layers to latent space
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, latent_dim)
        
    def forward(self, x):
        # Encoder forward pass
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        
        # Global average pooling and fully connected
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class Decoder(nn.Module):
    """Decoder part of the autoencoder"""
    
    def __init__(self, latent_dim=128, output_channels=3):
        super(Decoder, self).__init__()
        
        # Initial fully connected layer
        self.fc = nn.Linear(latent_dim, 1024 * 8 * 8)
        
        # Transposed convolutional layers
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)  # 8 -> 16
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 16 -> 32
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 32 -> 64
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 64 -> 128
        self.deconv5 = nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1)  # 128 -> 256
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
    def forward(self, x):
        # Reshape to feature maps
        x = self.fc(x)
        x = x.view(x.size(0), 1024, 8, 8)
        
        # Decoder forward pass
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = torch.sigmoid(self.deconv5(x))  # Output in [0, 1] range
        
        return x


class Autoencoder(nn.Module):
    """Complete autoencoder model"""
    
    def __init__(self, input_channels=3, latent_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x):
        """Get latent representation"""
        return self.encoder(x)
    
    def decode(self, latent):
        """Reconstruct from latent representation"""
        return self.decoder(latent) 