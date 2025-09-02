# Federated Autoencoder for Kidney Stone Classification

This project implements a federated learning framework using PyTorch and Flower to train an autoencoder on kidney stone image datasets. The autoencoder learns embeddings for different classes (MIX, SEC, SUR) from two datasets in a non-IID federated setting with configurable image corruptions.

## Features

- **Federated Learning**: Uses Flower framework for distributed training
- **Non-IID Data Distribution**: Implements Dirichlet distribution for realistic federated scenarios
- **Image Corruptions**: Configurable random corruptions (noise, blur, brightness, contrast)
- **Autoencoder Architecture**: Deep convolutional autoencoder for learning image embeddings
- **Multiple Datasets**: Supports Michel Daudon and Jonathan El-Beze kidney stone datasets
- **Flexible Configuration**: Easy-to-modify hyperparameters and settings

## Project Structure

```
federated-learning/
├── data/                           # Dataset directory
│   ├── Michel Daudon (w256 1k v1)/
│   │   ├── MIX/
│   │   │   ├── train/
│   │   │   └── test/
│   │   ├── SEC/
│   │   │   ├── train/
│   │   │   └── test/
│   │   └── SUR/
│   │       ├── train/
│   │       └── test/
│   └── Jonathan El-Beze (w256 1k v1)/
│       ├── MIX/
│       ├── SEC/
│       └── SUR/
├── models/
│   ├── __init__.py
│   └── autoencoder.py              # Autoencoder model definition
├── utils/
│   ├── __init__.py
│   └── data_utils.py               # Data loading and preprocessing utilities
├── config.py                       # Configuration file
├── client.py                       # Flower client implementation
├── server.py                       # Flower server implementation
├── run_client.py                   # Script to run individual clients
├── run_simulation.py               # Simulation script for testing
├── requirements.txt                # Python dependencies
└── README.md                       # This file
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

## Usage

### 1. Simulation Mode (Recommended for Testing)

Run the entire federated learning process in simulation mode:

```bash
# Basic simulation with all datasets and subversions
python run_simulation.py --save_results --plot_results

# Train only on specific dataset
python run_simulation.py --datasets "Michel Daudon (w256 1k v1)" --save_results --plot_results

# Train only on specific subversions
python run_simulation.py --subversions MIX SEC --save_results --plot_results

# Train on specific dataset and subversion
python run_simulation.py --datasets "Jonathan El-Beze (w256 1k v1)" --subversions SUR --save_results --plot_results

# Custom simulation with specific parameters
python run_simulation.py \
    --datasets "Michel Daudon (w256 1k v1)" "Jonathan El-Beze (w256 1k v1)" \
    --subversions MIX SEC \
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

## Image Corruptions

The framework supports various image corruptions:

1. **Gaussian Noise**: Random noise addition
2. **Salt & Pepper Noise**: Random pixel corruption
3. **Blur**: Gaussian blur with random radius
4. **Brightness**: Random brightness adjustment
5. **Contrast**: Random contrast modification

Each client applies random corruptions based on the specified probability.

## Results and Monitoring

The simulation script provides:

- **Training plots**: Loss curves over federated rounds
- **Configuration logs**: All hyperparameters and settings
- **Numerical results**: Best/final loss values
- **Model checkpoints**: Saved in `models/` directory

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

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{federated_autoencoder_kidney_stones,
  title={Federated Autoencoder for Kidney Stone Classification},
  author={Ivan Reyes-Amezcua},
  year={2025},
  howpublished={\url{https://github.com/Ivanrs297/federated-kidney-stone-corruption-detection}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 