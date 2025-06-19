# src/ml_models/pytorch_models.py

"""
PyTorch model implementations for the ML pipeline.

This module contains PyTorch implementations of the models used in the pipeline,
following the BaseModel interface defined in model_interface.py.
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.ml_models.model_interface import BaseModel, ModelFactory, ModelRegistry
from src.ml_models.model_config import ModelConfig

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


class EarlyStopping:
    """
    Early stopping implementation for PyTorch training.
    
    This class monitors a validation metric and stops training when the metric
    stops improving for a specified number of epochs.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0, monitor: str = "val_loss"):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            monitor (str): Metric to monitor
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state_dict = None
    
    def __call__(self, val_metrics: Dict[str, float], model: nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_metrics (Dict[str, float]): Validation metrics
            model (nn.Module): The model being trained
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        score = val_metrics.get(self.monitor, float('inf'))
        
        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = model.state_dict().copy()
        elif score > self.best_score + self.min_delta:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.best_state_dict = model.state_dict().copy()
            self.counter = 0
        
        return False
    
    def restore_best_weights(self, model: nn.Module) -> None:
        """
        Restore the best model weights.
        
        Args:
            model (nn.Module): The model to restore weights for
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            logging.info("Restored model to best weights")


class PyTorchLSTM(nn.Module):
    """
    PyTorch LSTM model for time series regression.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float, bidirectional: bool, fc_layers: List[int], 
                 activations: List[str]):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            bidirectional (bool): Whether to use bidirectional LSTM
            fc_layers (List[int]): Sizes of fully connected layers
            activations (List[str]): Activation functions for FC layers
        """
        super(PyTorchLSTM, self).__init__()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate the output size of the LSTM
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers
        fc_input_size = lstm_output_size
        fc_layers_list = []
        
        for i, (units, activation) in enumerate(zip(fc_layers, activations)):
            fc_layers_list.append(nn.Linear(fc_input_size, units))
            
            if activation == 'relu':
                fc_layers_list.append(nn.ReLU())
            elif activation == 'tanh':
                fc_layers_list.append(nn.Tanh())
            elif activation == 'sigmoid':
                fc_layers_list.append(nn.Sigmoid())
            # No activation for 'linear'
            
            fc_input_size = units
        
        self.fc_layers = nn.Sequential(*fc_layers_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc_layers(lstm_out)
        
        return output


class PyTorchAutoencoder(nn.Module):
    """
    PyTorch Autoencoder model for unsupervised feature learning.
    """
    
    def __init__(self, input_size: int, latent_dim: int, encoder_layers: List[int], 
                 decoder_layers: List[int], activations: List[str]):
        """
        Initialize the Autoencoder model.
        
        Args:
            input_size (int): Size of input features (flattened)
            latent_dim (int): Size of latent space
            encoder_layers (List[int]): Sizes of encoder layers
            decoder_layers (List[int]): Sizes of decoder layers
            activations (List[str]): Activation functions
        """
        super(PyTorchAutoencoder, self).__init__()
        
        # Process encoder layers
        encoder_layers_list = []
        current_size = input_size
        
        for i, units in enumerate(encoder_layers):
            # Handle special case for latent_dim
            if units == "latent_dim":
                units = latent_dim
                
            encoder_layers_list.append(nn.Linear(current_size, units))
            
            # Add activation (using the first activations)
            if i < len(activations):
                if activations[i] == 'relu':
                    encoder_layers_list.append(nn.ReLU())
                elif activations[i] == 'tanh':
                    encoder_layers_list.append(nn.Tanh())
                elif activations[i] == 'sigmoid':
                    encoder_layers_list.append(nn.Sigmoid())
            
            current_size = units
        
        self.encoder = nn.Sequential(*encoder_layers_list)
        
        # Process decoder layers
        decoder_layers_list = []
        current_size = latent_dim
        
        for i, units in enumerate(decoder_layers):
            # Handle special case for input_size
            if units == "input_size":
                units = input_size
                
            decoder_layers_list.append(nn.Linear(current_size, units))
            
            # Add activation (using the remaining activations)
            activation_idx = i + len(encoder_layers)
            if activation_idx < len(activations):
                if activations[activation_idx] == 'relu':
                    decoder_layers_list.append(nn.ReLU())
                elif activations[activation_idx] == 'tanh':
                    decoder_layers_list.append(nn.Tanh())
                elif activations[activation_idx] == 'sigmoid':
                    decoder_layers_list.append(nn.Sigmoid())
            
            current_size = units
        
        self.decoder = nn.Sequential(*decoder_layers_list)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            z (torch.Tensor): Latent representation
            
        Returns:
            torch.Tensor: Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Autoencoder model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Reconstructed input
        """
        # Flatten the input if it's not already flat
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.reshape(original_shape[0], -1)
            
        # Encode and decode
        z = self.encode(x)
        reconstruction = self.decode(z)
        
        # Reshape back to original shape if needed
        if len(original_shape) > 2:
            reconstruction = reconstruction.reshape(original_shape)
            
        return reconstruction


class PyTorchVAE(nn.Module):
    """
    PyTorch Variational Autoencoder (VAE) model.
    """
    
    def __init__(self, input_size: int, latent_dim: int, encoder_layers: List[int], 
                 decoder_layers: List[int], activations: List[str]):
        """
        Initialize the VAE model.
        
        Args:
            input_size (int): Size of input features (flattened)
            latent_dim (int): Size of latent space
            encoder_layers (List[int]): Sizes of encoder layers
            decoder_layers (List[int]): Sizes of decoder layers
            activations (List[str]): Activation functions
        """
        super(PyTorchVAE, self).__init__()
        
        # Process encoder layers
        encoder_layers_list = []
        current_size = input_size
        
        for i, units in enumerate(encoder_layers):
            encoder_layers_list.append(nn.Linear(current_size, units))
            
            # Add activation
            if i < len(activations):
                if activations[i] == 'relu':
                    encoder_layers_list.append(nn.ReLU())
                elif activations[i] == 'tanh':
                    encoder_layers_list.append(nn.Tanh())
                elif activations[i] == 'sigmoid':
                    encoder_layers_list.append(nn.Sigmoid())
            
            current_size = units
        
        self.encoder = nn.Sequential(*encoder_layers_list)
        
        # Mean and log variance layers
        self.fc_mu = nn.Linear(current_size, latent_dim)
        self.fc_log_var = nn.Linear(current_size, latent_dim)
        
        # Process decoder layers
        decoder_layers_list = []
        current_size = latent_dim
        
        for i, units in enumerate(decoder_layers):
            # Handle special case for input_size
            if units == "input_size":
                units = input_size
                
            decoder_layers_list.append(nn.Linear(current_size, units))
            
            # Add activation
            activation_idx = i + 1  # +1 because we start from the latent space
            if activation_idx < len(activations):
                if activations[activation_idx] == 'relu':
                    decoder_layers_list.append(nn.ReLU())
                elif activations[activation_idx] == 'tanh':
                    decoder_layers_list.append(nn.Tanh())
                elif activations[activation_idx] == 'sigmoid':
                    decoder_layers_list.append(nn.Sigmoid())
            
            current_size = units
        
        self.decoder = nn.Sequential(*decoder_layers_list)
        
        # Save for loss calculation
        self.input_size = input_size
        self.latent_dim = latent_dim
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and log variance of latent distribution
        """
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution.
        
        Args:
            mu (torch.Tensor): Mean of latent distribution
            log_var (torch.Tensor): Log variance of latent distribution
            
        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z (torch.Tensor): Latent vector
            
        Returns:
            torch.Tensor: Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the VAE model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                Reconstructed input, mean, and log variance
        """
        # Flatten the input if it's not already flat
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.reshape(original_shape[0], -1)
            
        # Encode
        mu, log_var = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode
        reconstruction = self.decode(z)
        
        # Reshape back to original shape if needed
        if len(original_shape) > 2:
            reconstruction = reconstruction.reshape(original_shape)
            
        return reconstruction, mu, log_var
    
    def loss_function(self, recon_x: torch.Tensor, x: torch.Tensor, 
                      mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        VAE loss function (reconstruction loss + KL divergence).
        
        Args:
            recon_x (torch.Tensor): Reconstructed input
            x (torch.Tensor): Original input
            mu (torch.Tensor): Mean of latent distribution
            log_var (torch.Tensor): Log variance of latent distribution
            
        Returns:
            torch.Tensor: Total loss
        """
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            recon_x = recon_x.reshape(recon_x.shape[0], -1)
            
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        return recon_loss + kl_loss


class PyTorchLSTMModel(BaseModel):
    """
    PyTorch LSTM model implementation of the BaseModel interface.
    """
    
    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the PyTorch LSTM model.
        
        Args:
            input_shape (Tuple[int, int]): Shape of the input data (window_size, features)
            config (Dict[str, Any]): Model configuration parameters
        """
        self.input_shape = input_shape
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract model parameters
        model_params = config.get("model_params", {})
        input_size = input_shape[1]  # Number of features
        hidden_size = model_params.get("hidden_size", 64)
        num_layers = model_params.get("num_layers", 2)
        dropout = model_params.get("dropout", 0.2)
        bidirectional = model_params.get("bidirectional", False)
        fc_layers = model_params.get("fc_layers", [32, 16, 1])
        activations = model_params.get("activations", ["relu", "relu", "linear"])
        
        # Create the model
        self.model = PyTorchLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            fc_layers=fc_layers,
            activations=activations
        ).to(self.device)
        
        # Set up optimizer
        optimizer_params = config.get("optimizer_params", {})
        optimizer_type = optimizer_params.get("type", "adam").lower()
        lr = optimizer_params.get("lr", 0.001)
        weight_decay = optimizer_params.get("weight_decay", 1e-5)
        
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            # Default to Adam
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        
        # Set up loss function
        loss_fn = config.get("loss_fn", "mse").lower()
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "mae" or loss_fn == "mean_absolute_error":
            self.criterion = nn.L1Loss()
        else:
            # Default to MSE
            self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the given data.
        
        Args:
            X (np.ndarray): Input features of shape (samples, window_size, features)
            y (np.ndarray): Target values of shape (samples,)
            **kwargs: Additional arguments for training
            
        Returns:
            Dict[str, Any]: Training history
        """
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Add dimension for output
        
        # Create dataset and data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Get training parameters
        train_params = self.config.get("train_params", {})
        batch_size = train_params.get("batch_size", 32)
        epochs = train_params.get("epochs", 100)
        validation_split = train_params.get("validation_split", 0.2)
        
        # Split dataset into training and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Set up early stopping
        early_stopping_params = train_params.get("early_stopping", {})
        patience = early_stopping_params.get("patience", 10)
        monitor = early_stopping_params.get("monitor", "val_loss")
        early_stopping = EarlyStopping(patience=patience, monitor=monitor)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # Move data to device
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    # Move data to device
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            # Log progress
            logging.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            
            # Check early stopping
            if early_stopping({"val_loss": val_loss}, self.model):
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best weights
        early_stopping.restore_best_weights(self.model)
        
        return self.history
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the model.
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional arguments for prediction
            
        Returns:
            np.ndarray: Predictions
        """
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions.flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray): Target values
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X_tensor)
            mse_loss = nn.functional.mse_loss(predictions, y_tensor).item()
            mae_loss = nn.functional.l1_loss(predictions, y_tensor).item()
        
        return {
            "mse": mse_loss,
            "mae": mae_loss,
            "rmse": np.sqrt(mse_loss)
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to the given path.
        
        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state dict, optimizer state dict, and config
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "input_shape": self.input_shape,
            "history": self.history
        }, path)
        
        logging.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PyTorchLSTMModel':
        """
        Load a model from the given path.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            PyTorchLSTMModel: Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # Create model instance
        model = cls(
            input_shape=checkpoint["input_shape"],
            config=checkpoint["config"]
        )
        
        # Load state dicts
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load history if available
        if "history" in checkpoint:
            model.history = checkpoint["history"]
        
        logging.info(f"Model loaded from {path}")
        return model


class PyTorchAutoencoderModel(BaseModel):
    """
    PyTorch Autoencoder model implementation of the BaseModel interface.
    """
    
    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the PyTorch Autoencoder model.
        
        Args:
            input_shape (Tuple[int, int]): Shape of the input data (window_size, features)
            config (Dict[str, Any]): Model configuration parameters
        """
        self.input_shape = input_shape
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate flattened input size
        input_size = input_shape[0] * input_shape[1]
        
        # Extract model parameters
        model_params = config.get("model_params", {})
        latent_dim = model_params.get("latent_dim", 32)
        encoder_layers = model_params.get("encoder_layers", [128, "latent_dim"])
        decoder_layers = model_params.get("decoder_layers", [128, "input_size"])
        activations = model_params.get("activations", ["relu", "relu", "relu", "sigmoid"])
        
        # Create the model
        self.model = PyTorchAutoencoder(
            input_size=input_size,
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activations=activations
        ).to(self.device)
        
        # Set up optimizer
        optimizer_params = config.get("optimizer_params", {})
        optimizer_type = optimizer_params.get("type", "adam").lower()
        lr = optimizer_params.get("lr", 0.001)
        weight_decay = optimizer_params.get("weight_decay", 1e-5)
        
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            # Default to Adam
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        
        # Set up loss function
        loss_fn = config.get("loss_fn", "mse").lower()
        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        elif loss_fn == "mae" or loss_fn == "mean_absolute_error":
            self.criterion = nn.L1Loss()
        else:
            # Default to MSE
            self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the given data.
        
        For autoencoders, y is ignored as the model learns to reconstruct X.
        
        Args:
            X (np.ndarray): Input features of shape (samples, window_size, features)
            y (np.ndarray, optional): Ignored for autoencoders
            **kwargs: Additional arguments for training
            
        Returns:
            Dict[str, Any]: Training history
        """
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Create dataset and data loader
        dataset = TensorDataset(X_tensor, X_tensor)  # X is both input and target
        
        # Get training parameters
        train_params = self.config.get("train_params", {})
        batch_size = train_params.get("batch_size", 32)
        epochs = train_params.get("epochs", 100)
        validation_split = train_params.get("validation_split", 0.2)
        
        # Split dataset into training and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Set up early stopping
        early_stopping_params = train_params.get("early_stopping", {})
        patience = early_stopping_params.get("patience", 10)
        monitor = early_stopping_params.get("monitor", "val_loss")
        early_stopping = EarlyStopping(patience=patience, monitor=monitor)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, _ in train_loader:
                # Move data to device
                batch_X = batch_X.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_X)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, _ in val_loader:
                    # Move data to device
                    batch_X = batch_X.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_X)
                    
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            # Log progress
            logging.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            
            # Check early stopping
            if early_stopping({"val_loss": val_loss}, self.model):
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best weights
        early_stopping.restore_best_weights(self.model)
        
        return self.history
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions (reconstructions) with the model.
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional arguments for prediction
            
        Returns:
            np.ndarray: Reconstructed inputs
        """
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            reconstructions = self.model(X_tensor).cpu().numpy()
        
        return reconstructions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        For autoencoders, y is ignored as the model is evaluated on reconstruction error.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray, optional): Ignored for autoencoders
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            mse_loss = nn.functional.mse_loss(reconstructions, X_tensor).item()
            mae_loss = nn.functional.l1_loss(reconstructions, X_tensor).item()
        
        return {
            "mse": mse_loss,
            "mae": mae_loss,
            "rmse": np.sqrt(mse_loss)
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to the given path.
        
        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state dict, optimizer state dict, and config
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "input_shape": self.input_shape,
            "history": self.history
        }, path)
        
        logging.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PyTorchAutoencoderModel':
        """
        Load a model from the given path.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            PyTorchAutoencoderModel: Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # Create model instance
        model = cls(
            input_shape=checkpoint["input_shape"],
            config=checkpoint["config"]
        )
        
        # Load state dicts
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load history if available
        if "history" in checkpoint:
            model.history = checkpoint["history"]
        
        logging.info(f"Model loaded from {path}")
        return model


class PyTorchVAEModel(BaseModel):
    """
    PyTorch Variational Autoencoder (VAE) model implementation of the BaseModel interface.
    """
    
    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the PyTorch VAE model.
        
        Args:
            input_shape (Tuple[int, int]): Shape of the input data (window_size, features)
            config (Dict[str, Any]): Model configuration parameters
        """
        self.input_shape = input_shape
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate flattened input size
        input_size = input_shape[0] * input_shape[1]
        
        # Extract model parameters
        model_params = config.get("model_params", {})
        latent_dim = model_params.get("latent_dim", 32)
        encoder_layers = model_params.get("encoder_layers", [128])
        decoder_layers = model_params.get("decoder_layers", [128, "input_size"])
        activations = model_params.get("activations", ["relu", "relu", "sigmoid"])
        
        # Create the model
        self.model = PyTorchVAE(
            input_size=input_size,
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            activations=activations
        ).to(self.device)
        
        # Set up optimizer
        optimizer_params = config.get("optimizer_params", {})
        optimizer_type = optimizer_params.get("type", "adam").lower()
        lr = optimizer_params.get("lr", 0.001)
        weight_decay = optimizer_params.get("weight_decay", 1e-5)
        
        if optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            # Default to Adam
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": []
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the given data.
        
        For VAEs, y is ignored as the model learns to reconstruct X.
        
        Args:
            X (np.ndarray): Input features of shape (samples, window_size, features)
            y (np.ndarray, optional): Ignored for VAEs
            **kwargs: Additional arguments for training
            
        Returns:
            Dict[str, Any]: Training history
        """
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Create dataset and data loader
        dataset = TensorDataset(X_tensor, X_tensor)  # X is both input and target
        
        # Get training parameters
        train_params = self.config.get("train_params", {})
        batch_size = train_params.get("batch_size", 32)
        epochs = train_params.get("epochs", 100)
        validation_split = train_params.get("validation_split", 0.2)
        
        # Split dataset into training and validation
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Set up early stopping
        early_stopping_params = train_params.get("early_stopping", {})
        patience = early_stopping_params.get("patience", 10)
        monitor = early_stopping_params.get("monitor", "val_loss")
        early_stopping = EarlyStopping(patience=patience, monitor=monitor)
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, _ in train_loader:
                # Move data to device
                batch_X = batch_X.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                recon_X, mu, log_var = self.model(batch_X)
                loss = self.model.loss_function(recon_X, batch_X, mu, log_var)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, _ in val_loader:
                    # Move data to device
                    batch_X = batch_X.to(self.device)
                    
                    # Forward pass
                    recon_X, mu, log_var = self.model(batch_X)
                    loss = self.model.loss_function(recon_X, batch_X, mu, log_var)
                    
                    val_loss += loss.item() * batch_X.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            # Log progress
            logging.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            
            # Check early stopping
            if early_stopping({"val_loss": val_loss}, self.model):
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Restore best weights
        early_stopping.restore_best_weights(self.model)
        
        return self.history
    
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions (reconstructions) with the model.
        
        Args:
            X (np.ndarray): Input features
            **kwargs: Additional arguments for prediction
            
        Returns:
            np.ndarray: Reconstructed inputs
        """
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            reconstructions, _, _ = self.model(X_tensor)
            reconstructions = reconstructions.cpu().numpy()
        
        return reconstructions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        For VAEs, y is ignored as the model is evaluated on reconstruction error.
        
        Args:
            X (np.ndarray): Input features
            y (np.ndarray, optional): Ignored for VAEs
            **kwargs: Additional arguments for evaluation
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Convert numpy array to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            reconstructions, mu, log_var = self.model(X_tensor)
            
            # Calculate losses
            vae_loss = self.model.loss_function(reconstructions, X_tensor, mu, log_var).item()
            mse_loss = nn.functional.mse_loss(reconstructions, X_tensor).item()
            mae_loss = nn.functional.l1_loss(reconstructions, X_tensor).item()
        
        return {
            "vae_loss": vae_loss,
            "mse": mse_loss,
            "mae": mae_loss,
            "rmse": np.sqrt(mse_loss)
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to the given path.
        
        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state dict, optimizer state dict, and config
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "input_shape": self.input_shape,
            "history": self.history
        }, path)
        
        logging.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PyTorchVAEModel':
        """
        Load a model from the given path.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            PyTorchVAEModel: Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # Create model instance
        model = cls(
            input_shape=checkpoint["input_shape"],
            config=checkpoint["config"]
        )
        
        # Load state dicts
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load history if available
        if "history" in checkpoint:
            model.history = checkpoint["history"]
        
        logging.info(f"Model loaded from {path}")
        return model


# --- Model Factories ---
class PyTorchLSTMFactory(ModelFactory):
    """Factory for creating PyTorch LSTM models."""
    
    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a PyTorch LSTM model."""
        return PyTorchLSTMModel(input_shape, config)


class PyTorchAutoencoderFactory(ModelFactory):
    """Factory for creating PyTorch Autoencoder models."""
    
    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a PyTorch Autoencoder model."""
        return PyTorchAutoencoderModel(input_shape, config)


class PyTorchVAEFactory(ModelFactory):
    """Factory for creating PyTorch VAE models."""
    
    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a PyTorch VAE model."""
        return PyTorchVAEModel(input_shape, config)


# Register factories with the model registry
ModelRegistry.register("pytorch_lstm", PyTorchLSTMFactory())
ModelRegistry.register("pytorch_autoencoder", PyTorchAutoencoderFactory())
ModelRegistry.register("pytorch_vae", PyTorchVAEFactory())
# Register aliases for backward compatibility
ModelRegistry.register("lstm", PyTorchLSTMFactory())
ModelRegistry.register("autoencoder", PyTorchAutoencoderFactory())
ModelRegistry.register("vae", PyTorchVAEFactory())