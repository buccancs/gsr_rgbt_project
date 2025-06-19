# src/ml_models/pytorch_resnet_models.py

"""
PyTorch ResNet model implementations for the ML pipeline.

This module contains PyTorch implementations of ResNet-based models used in the pipeline,
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


class ResidualBlock(nn.Module):
    """
    Basic residual block for ResNet.
    
    This block consists of two convolutional layers with batch normalization and a skip connection.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        """
        Initialize the residual block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            stride (int): Stride for the first convolution
            downsample (nn.Module, optional): Downsampling layer for the skip connection
        """
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Downsampling layer for skip connection (if needed)
        self.downsample = downsample
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        identity = x
        
        # First conv layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv layer
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsampling to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        
        return out


class PyTorchResNet(nn.Module):
    """
    PyTorch ResNet model for time series regression.
    
    This model uses residual blocks to process time series data and make predictions.
    """
    
    def __init__(self, input_channels: int, layers: List[int], blocks_per_layer: List[int],
                 fc_layers: List[int], activations: List[str], dropout_rate: float = 0.2):
        """
        Initialize the ResNet model.
        
        Args:
            input_channels (int): Number of input channels (features)
            layers (List[int]): Number of channels for each layer
            blocks_per_layer (List[int]): Number of residual blocks per layer
            fc_layers (List[int]): Sizes of fully connected layers
            activations (List[str]): Activation functions for FC layers
            dropout_rate (float): Dropout rate for FC layers
        """
        super(PyTorchResNet, self).__init__()
        
        # Initial convolutional layer
        self.in_channels = 64
        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(layers[0], blocks_per_layer[0], stride=1)
        self.layer2 = self._make_layer(layers[1], blocks_per_layer[1], stride=2)
        self.layer3 = self._make_layer(layers[2], blocks_per_layer[2], stride=2)
        self.layer4 = self._make_layer(layers[3], blocks_per_layer[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        fc_layers_list = []
        current_size = layers[3]  # Output channels of the last residual layer
        
        for i, (units, activation) in enumerate(zip(fc_layers, activations)):
            fc_layers_list.append(nn.Linear(current_size, units))
            
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
            if i < len(fc_layers) - 1 and dropout_rate > 0:
                fc_layers_list.append(nn.Dropout(dropout_rate))
            
            current_size = units
        
        self.fc_layers = nn.Sequential(*fc_layers_list)
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """
        Create a layer of residual blocks.
        
        Args:
            out_channels (int): Number of output channels
            blocks (int): Number of residual blocks
            stride (int): Stride for the first block
            
        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        downsample = None
        
        # Create downsampling layer if needed
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        
        # First block may have a different stride
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels
        
        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Reshape input to (batch_size, channels, length) for 1D convolution
        x = x.permute(0, 2, 1)  # (batch_size, features, seq_len)
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        return x


class PyTorchResNetModel(BaseModel):
    """
    PyTorch ResNet model implementation of the BaseModel interface.
    """
    
    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the PyTorch ResNet model.
        
        Args:
            input_shape (Tuple[int, int]): Shape of the input data (window_size, features)
            config (Dict[str, Any]): Model configuration parameters
        """
        self.input_shape = input_shape
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract model parameters
        model_params = config.get("model_params", {})
        input_channels = input_shape[1]  # Number of features
        
        # ResNet architecture parameters
        layers = model_params.get("layers", [64, 128, 256, 512])
        blocks_per_layer = model_params.get("blocks_per_layer", [2, 2, 2, 2])  # ResNet-18 by default
        fc_layers = model_params.get("fc_layers", [256, 64, 1])
        activations = model_params.get("activations", ["relu", "relu", "linear"])
        dropout_rate = model_params.get("dropout_rate", 0.2)
        
        # Create the model
        self.model = PyTorchResNet(
            input_channels=input_channels,
            layers=layers,
            blocks_per_layer=blocks_per_layer,
            fc_layers=fc_layers,
            activations=activations,
            dropout_rate=dropout_rate
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
    def load(cls, path: str) -> 'PyTorchResNetModel':
        """
        Load a model from the given path.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            PyTorchResNetModel: Loaded model
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
class PyTorchResNetFactory(ModelFactory):
    """Factory for creating PyTorch ResNet models."""
    
    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a PyTorch ResNet model."""
        return PyTorchResNetModel(input_shape, config)


# Register factory with the model registry
ModelRegistry.register("pytorch_resnet", PyTorchResNetFactory())
# Register alias for convenience
ModelRegistry.register("resnet", PyTorchResNetFactory())