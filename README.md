# FedAgain: Federated Kidney Stone Corruption Detection

A federated learning framework for training robust autoencoder models to detect image corruptions in medical kidney stone datasets. The system evaluates model performance against 16 different types of image corruptions while preserving data privacy through distributed training.

## 🔬 Features

- **Federated Learning**: Uses Flower framework for distributed training across multiple clients
- **Corruption Detection**: Robust evaluation against 16 corruption types (blur, noise, brightness, contrast, etc.)
- **Medical Image Analysis**: Specialized for kidney stone classification (MIX, SEC, SUR types)
- **Non-IID Data Distribution**: Implements Dirichlet distribution for realistic federated scenarios
- **Autoencoder Architecture**: Deep convolutional autoencoder for anomaly detection
- **Multi-Dataset Support**: Compatible with Jonathan El-Beze, Michel Daudon, and MyStone datasets
- **Comprehensive Metrics**: ROC-AUC, precision, recall, F1-score evaluation
- **Robustness Testing**: Systematic evaluation of model performance under various corruptions

## Project Structure

```
federated-kidney-stone-corruption-detection/
├── data/                           # Dataset directory
│   ├── Michel Daudon (w256 1k v1)/
│   │   ├── MIX/, SEC/, SUR/        # 6 kidney stone subtypes each
│   │   │   ├── train/              # Training images
│   │   │   └── test/               # Test images
│   ├── Jonathan El-Beze (w256 1k v1)/
│   │   ├── MIX/, SEC/, SUR/        # Same structure
│   └── MyStone (w256, 1k)/
│       ├── train/                  # 4,800 JPG images
│       └── test/                   # 1,200 JPG images
├── endoscopycorruptions/           # Corruption functions
│   ├── __init__.py
│   └── corruptions.py              # 16 corruption implementations
├── models/
│   ├── __init__.py
│   └── autoencoder.py              # Deep convolutional autoencoder
├── utils/
│   ├── __init__.py
│   ├── data_utils.py               # Data loading utilities
│   └── metrics.py                  # Comprehensive evaluation metrics
├── robust_divergence_experiments/
│   └── train.py                    # Main training script
├── config.py                       # Configuration parameters
├── client.py                       # Federated client implementation
├── server.py                       # Federated server strategy
├── run_simulation.py               # Federated simulation runner
├── requirements.txt                # Python dependencies
└── README.md                       # This documentation
```

## Installation

1. **Clone the repository** (or ensure you're in the project directory):
   ```bash
   cd federated-learning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify data structure**: Ensure your `data/` folder contains the kidney stone datasets with the expected structure.

## 🚀 Quick Start

### Basic Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run main training script with corruption detection
python robust_divergence_experiments/train.py

# Run federated simulation (recommended)
python run_simulation.py --save_results --plot_results
```

### Advanced Usage

```bash
# Train on specific dataset
python run_simulation.py --datasets "Michel Daudon (w256 1k v1)" --save_results

# Train on specific kidney stone types
python run_simulation.py --subversions MIX SEC --save_results --plot_results

# Custom federated parameters
python run_simulation.py \
    --datasets "Michel Daudon (w256 1k v1)" "Jonathan El-Beze (w256 1k v1)" \
    --subversions MIX SEC SUR \
    --num_clients 8 \
    --num_rounds 15 \
    --corruption_prob 0.2 \
    --alpha 0.3 \
    --clients_per_round 5 \
    --save_results \
    --plot_results
```

### 2. Distributed Mode (Multiple Processes)

For real distributed training, run the server and clients separately:

**Start the server:**
```bash
# Server with all datasets and subversions
python server.py

# Server with specific datasets/subversions
python server.py --datasets "Michel Daudon (w256 1k v1)" --subversions MIX SEC
```

**Start clients** (in separate terminals):
```bash
# Client 0 - all datasets and subversions
python run_client.py --client_id 0 --num_clients 5 --corruption_prob 0.1

# Client 1 - specific dataset
python run_client.py --client_id 1 --datasets "Jonathan El-Beze (w256 1k v1)" --corruption_prob 0.15

# Client 2 - specific subversions
python run_client.py --client_id 2 --subversions MIX SUR --corruption_prob 0.2

# ... and so on for each client
```

## Configuration

Modify `config.py` to adjust hyperparameters:

```python
# Federated Learning Configuration
NUM_CLIENTS = 5                     # Number of clients
NUM_ROUNDS = 10                     # Number of federated rounds
CLIENTS_PER_ROUND = 3               # Clients participating per round

# Model Configuration
LATENT_DIM = 128                    # Autoencoder latent dimension
LEARNING_RATE = 0.001               # Learning rate
BATCH_SIZE = 32                     # Batch size
LOCAL_EPOCHS = 5                    # Local epochs per client

# Data Configuration
CORRUPTION_PROBABILITY = 0.1        # Image corruption probability
ALPHA = 0.5                         # Non-IID distribution parameter
```

## Key Parameters

### Non-IID Distribution (`alpha`)
- **Lower values (0.1-0.3)**: More heterogeneous data distribution
- **Higher values (0.8-1.0)**: More homogeneous data distribution
- **Default: 0.5**: Moderate heterogeneity

### Corruption Probability
- **0.0**: No image corruptions
- **0.1**: Light corruptions (recommended)
- **0.3+**: Heavy corruptions (challenging scenario)

### Number of Clients
- Adjust based on your computational resources
- More clients = more realistic federated scenario
- Each client gets a subset of the data based on non-IID distribution

## Model Architecture

The autoencoder consists of:

**Encoder:**
- 5 convolutional layers with batch normalization
- Progressive downsampling (256×256 → 8×8)
- Global average pooling + fully connected layer
- Output: 128-dimensional latent representation

**Decoder:**
- Fully connected layer + reshape
- 5 transposed convolutional layers
- Progressive upsampling (8×8 → 256×256)
- Output: Reconstructed 256×256×3 image

## 🔧 Corruption Types

The system evaluates robustness against 16 different corruption types:

### **Visual Corruptions:**
- **Gaussian Noise**: Random noise addition
- **Shot Noise**: Poisson noise simulation  
- **Impulse Noise**: Salt & pepper corruption
- **Speckle Noise**: Multiplicative noise

### **Blur Corruptions:**
- **Defocus Blur**: Out-of-focus simulation
- **Motion Blur**: Camera shake effects
- **Zoom Blur**: Radial blur patterns

### **Weather Corruptions:**
- **Fog**: Atmospheric haze simulation
- **Frost**: Ice crystal patterns
- **Snow**: Snowfall effects

### **Digital Corruptions:**
- **JPEG Compression**: Compression artifacts
- **Pixelate**: Resolution reduction
- **Elastic Transform**: Geometric distortions

### **Lighting Corruptions:**
- **Brightness**: Illumination changes
- **Contrast**: Dynamic range modification
- **Saturate**: Color saturation effects

Each corruption is tested at 5 severity levels for comprehensive robustness evaluation.

## 📊 Dataset Statistics

### **Total Dataset Size**: ~50,000+ Images

| Dataset | MIX | SEC | SUR | Total |
|---------|-----|-----|-----|-------|
| **Jonathan El-Beze** | 12,000 | 6,000 | 6,000 | 24,000 |
| **Michel Daudon** | 12,000 | 6,000 | 6,023 | 24,023 |
| **MyStone** | - | - | - | 6,000 |

### **Kidney Stone Subtypes** (6 classes):
- **Type Ia**: Whewellite monohydrate
- **Type IIa**: Whewellite dihydrate  
- **Type IIb**: Weddellite
- **Type IVa**: Uric acid
- **Type IVd**: Uric acid dihydrate
- **Type VIa**: Brushite

## 📈 Results and Monitoring

The system provides comprehensive evaluation:

- **Corruption Robustness**: ROC-AUC scores across all 16 corruptions
- **Training Curves**: Loss progression over federated rounds
- **Performance Metrics**: Precision, recall, F1-score for each corruption type
- **Model Checkpoints**: Best models saved with full metrics
- **Visualization**: Corruption examples and detection results

## Example Output

```
============================================================
FEDERATED AUTOENCODER TRAINING SIMULATION
============================================================
Setting up simulation...
Number of clients: 5
Corruption probability: 0.1
Non-IID alpha: 0.5
Total images loaded: 12000
Unique labels: {'Michel Daudon (w256 1k v1)_MIX', 'Michel Daudon (w256 1k v1)_SEC', ...}
Created 5 client datasets
Client 0: 2400 samples
Client 1: 2200 samples
...

Starting simulation with 5 clients for 10 rounds...
Round 1: Global test loss = 0.245631
Round 2: Global test loss = 0.198432
...
============================================================
TRAINING COMPLETED
============================================================
Final global test loss: 0.087234
Best global test loss: 0.085123 (Round 8)
```

## Troubleshooting

1. **CUDA out of memory**: Reduce `BATCH_SIZE` in `config.py`
2. **Too few samples per client**: Increase `NUM_CLIENTS` or decrease `MIN_SAMPLES_PER_CLIENT`
3. **Slow training**: Reduce `LOCAL_EPOCHS` or `NUM_ROUNDS`
4. **Data loading errors**: Verify the data directory structure matches the expected format

## Advanced Usage

### Custom Corruption Functions

Add new corruption types in `utils/data_utils.py`:

```python
def custom_corruption(self, image):
    """Your custom corruption function"""
    if random.random() < self.corruption_prob:
        # Apply your corruption
        pass
    return image
```

### Different Model Architectures

Modify `models/autoencoder.py` to experiment with different architectures:

- Change the number of layers
- Adjust the latent dimension
- Add skip connections
- Implement variational autoencoders

### Custom Aggregation Strategies

Extend `server.py` to implement different federated learning strategies:

- FedProx
- FedNova
- Custom weighted averaging

## 🏥 Medical Applications

This framework is designed for:

- **Robust Medical Image Analysis**: Ensuring model reliability under real-world image quality variations
- **Privacy-Preserving Healthcare**: Training on distributed medical data without sharing sensitive information
- **Quality Assurance**: Detecting image corruptions that could affect diagnostic accuracy
- **Cross-Institution Collaboration**: Enabling multi-hospital research without data centralization

## 🔬 Research Applications

- **Federated Learning Research**: Benchmarking FL algorithms on medical data
- **Robustness Evaluation**: Systematic testing of model reliability
- **Medical AI Safety**: Ensuring consistent performance across imaging conditions
- **Domain Adaptation**: Studying performance across different medical imaging setups

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{fedagai_federated_kidney_stone_corruption_detection,
  title={FedAgain - Federated Learning for Robust Kidney Stone Corruption Detection},
  author={Ivan Reyes-Amezcua},
  year={2025},
  howpublished={\url{https://github.com/Ivanrs297/federated-kidney-stone-corruption-detection}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 