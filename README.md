# Adversarial Autoencoder

This project implements an adversarial autoencoder for image generation. The pipeline consists of three main components:

1. **Data Preparation (`data.py`):**
   - Creates the dataset and dataloaders for training.
   - Utilizes the configuration parameters defined in `config.json`.

2. **Model Architecture (`model.py`):**
   - Defines the architecture for the adversarial autoencoder using PyTorch's `nn.Module`.
   - Architecture details and hyperparameters are configured in `config.json`.

3. **Training Loop (`train.py`):**
   - Executes the training loop for the adversarial autoencoder.
   - Loads data, initializes the model, and optimizes using specified parameters.
