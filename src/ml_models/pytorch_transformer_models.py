# src/ml_models/pytorch_transformer_models.py

"""
PyTorch Transformer model implementations for the ML pipeline.

This module contains PyTorch implementations of Transformer-based models used in the pipeline,
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
from src.ml_models.pytorch_models import EarlyStopping

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer model.
    
    This adds positional information to the input embeddings to help the model
    understand the sequence order.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model (int): The embedding dimension
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class PyTorchTransformer(nn.Module):
    """
    PyTorch Transformer model for time series regression.
    
    This model uses a Transformer encoder to process time series data and make predictions.
    """
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float, fc_layers: List[int],
                 activations: List[str]):
        """
        Initialize the Transformer model.
        
        Args:
            input_size (int): Size of input features
            d_model (int): Dimension of the model (must be divisible by nhead)
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Dimension of the feedforward network
            dropout (float): Dropout rate
            fc_layers (List[int]): Sizes of fully connected layers
            activations (List[str]): Activation functions for FC layers
        """
        super(PyTorchTransformer, self).__init__()
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Fully connected layers
        fc_input_size = d_model
        fc_layers_list = []
        
        for i, (units, activation) in enumerate(zip(fc_layers, activations)):
            fc_layers_list.append(nn.Linear(fc_input_size, units))
            
            if activation == 'relu':
                fc_layers_list.append(nn.ReLU())
            elif activation == 'tanh':
                fc_layers_list.append(nn.Tanh())
            elif activation == 'sigmoid':
                fc_layers_list.append(nn.Sigmoid())
            elif activation == 'leaky_relu':
                fc_layers_list.append(nn.LeakyReLU(0.1))
            # No activation for 'linear'
            
            # Add dropout after activation (except for the last layer)
            if i < len(fc_layers) - 1 and dropout > 0:
                fc_layers_list.append(nn.Dropout(dropout))
            
            fc_input_size = units
        
        self.fc_layers = nn.Sequential(*fc_layers_list)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the output from the last time step
        x = x[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc_layers(x)
        
        return output


class PyTorchTransformerModel(BaseModel):
    """
    PyTorch Transformer model implementation of the BaseModel interface.
    """
    
    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the PyTorch Transformer model.
        
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
        d_model = model_params.get("d_model", 64)
        nhead = model_params.get("nhead", 4)
        num_layers = model_params.get("num_layers", 2)
        dim_feedforward = model_params.get("dim_feedforward", 256)
        dropout = model_params.get("dropout", 0.1)
        fc_layers = model_params.get("fc_layers", [32, 16, 1])
        activations = model_params.get("activations", ["relu", "relu", "linear"])
        
        # Create the model
        self.model = PyTorchTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
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
    def load(cls, path: str) -> 'PyTorchTransformerModel':
        """
        Load a model from the given path.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            PyTorchTransformerModel: Loaded model
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


# --- Model Factory ---
class PyTorchTransformerFactory(ModelFactory):
    """Factory for creating PyTorch Transformer models."""
    
    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a PyTorch Transformer model."""
        return PyTorchTransformerModel(input_shape, config)


# Register factory with the model registry
ModelRegistry.register("pytorch_transformer", PyTorchTransformerFactory())
# Register alias for convenience
ModelRegistry.register("transformer", PyTorchTransformerFactory())