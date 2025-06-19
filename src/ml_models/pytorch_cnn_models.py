# src/ml_models/pytorch_cnn_models.py

"""
PyTorch CNN model implementations for the ML pipeline.

This module contains PyTorch implementations of CNN-based models used in the pipeline,
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


class PyTorchCNN(nn.Module):
    """
    PyTorch CNN model for time series regression.
    
    This model treats the time series data as a 1D signal and applies
    1D convolutions to extract features before making predictions.
    """
    
    def __init__(self, input_channels: int, input_length: int, 
                 conv_channels: List[int], kernel_sizes: List[int], 
                 strides: List[int], fc_layers: List[int], 
                 activations: List[str], pool_sizes: List[int] = None,
                 dropout_rate: float = 0.2):
        """
        Initialize the CNN model.
        
        Args:
            input_channels (int): Number of input channels (features)
            input_length (int): Length of the input sequence
            conv_channels (List[int]): Number of channels for each conv layer
            kernel_sizes (List[int]): Kernel sizes for each conv layer
            strides (List[int]): Strides for each conv layer
            fc_layers (List[int]): Sizes of fully connected layers
            activations (List[str]): Activation functions for each layer
            pool_sizes (List[int], optional): Pooling sizes for each conv layer
            dropout_rate (float, optional): Dropout rate for FC layers
        """
        super(PyTorchCNN, self).__init__()
        
        # Validate input parameters
        assert len(conv_channels) == len(kernel_sizes) == len(strides), \
            "conv_channels, kernel_sizes, and strides must have the same length"
        
        if pool_sizes is None:
            pool_sizes = [2] * len(conv_channels)
        else:
            assert len(pool_sizes) == len(conv_channels), \
                "pool_sizes must have the same length as conv_channels"
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        current_channels = input_channels
        current_length = input_length
        
        for i, (out_channels, kernel_size, stride, pool_size) in enumerate(
            zip(conv_channels, kernel_sizes, strides, pool_sizes)
        ):
            # Add convolutional layer
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2  # Same padding
                )
            )
            
            # Calculate output size after convolution
            current_length = (current_length + 2 * (kernel_size // 2) - kernel_size) // stride + 1
            
            # Add activation
            if i < len(activations):
                if activations[i] == 'relu':
                    self.conv_layers.append(nn.ReLU())
                elif activations[i] == 'tanh':
                    self.conv_layers.append(nn.Tanh())
                elif activations[i] == 'sigmoid':
                    self.conv_layers.append(nn.Sigmoid())
                elif activations[i] == 'leaky_relu':
                    self.conv_layers.append(nn.LeakyReLU(0.1))
            
            # Add pooling layer
            if pool_size > 1:
                self.conv_layers.append(nn.MaxPool1d(pool_size))
                current_length = current_length // pool_size
            
            # Update current channels for next layer
            current_channels = out_channels
        
        # Calculate flattened size after convolutions
        self.flattened_size = current_channels * current_length
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        current_size = self.flattened_size
        
        for i, units in enumerate(fc_layers):
            self.fc_layers.append(nn.Linear(current_size, units))
            
            # Add activation (using the remaining activations)
            activation_idx = i + len(conv_channels)
            if activation_idx < len(activations):
                if activations[activation_idx] == 'relu':
                    self.fc_layers.append(nn.ReLU())
                elif activations[activation_idx] == 'tanh':
                    self.fc_layers.append(nn.Tanh())
                elif activations[activation_idx] == 'sigmoid':
                    self.fc_layers.append(nn.Sigmoid())
                elif activations[activation_idx] == 'leaky_relu':
                    self.fc_layers.append(nn.LeakyReLU(0.1))
            
            # Add dropout after activation (except for the last layer)
            if i < len(fc_layers) - 1 and dropout_rate > 0:
                self.fc_layers.append(nn.Dropout(dropout_rate))
            
            current_size = units
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Reshape input to (batch_size, channels, length) for 1D convolution
        x = x.permute(0, 2, 1)  # (batch_size, features, seq_len)
        
        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Pass through fully connected layers
        for layer in self.fc_layers:
            x = layer(x)
        
        return x


class PyTorchCNNLSTM(nn.Module):
    """
    PyTorch CNN-LSTM hybrid model for time series regression.
    
    This model first applies CNN layers to extract features from the time series,
    then passes these features to an LSTM to capture temporal dependencies.
    """
    
    def __init__(self, input_channels: int, input_length: int, 
                 conv_channels: List[int], kernel_sizes: List[int], 
                 strides: List[int], lstm_hidden_size: int, 
                 lstm_num_layers: int, fc_layers: List[int], 
                 activations: List[str], pool_sizes: List[int] = None,
                 dropout_rate: float = 0.2, bidirectional: bool = False):
        """
        Initialize the CNN-LSTM model.
        
        Args:
            input_channels (int): Number of input channels (features)
            input_length (int): Length of the input sequence
            conv_channels (List[int]): Number of channels for each conv layer
            kernel_sizes (List[int]): Kernel sizes for each conv layer
            strides (List[int]): Strides for each conv layer
            lstm_hidden_size (int): Size of LSTM hidden state
            lstm_num_layers (int): Number of LSTM layers
            fc_layers (List[int]): Sizes of fully connected layers
            activations (List[str]): Activation functions for each layer
            pool_sizes (List[int], optional): Pooling sizes for each conv layer
            dropout_rate (float, optional): Dropout rate
            bidirectional (bool, optional): Whether to use bidirectional LSTM
        """
        super(PyTorchCNNLSTM, self).__init__()
        
        # Validate input parameters
        assert len(conv_channels) == len(kernel_sizes) == len(strides), \
            "conv_channels, kernel_sizes, and strides must have the same length"
        
        if pool_sizes is None:
            pool_sizes = [2] * len(conv_channels)
        else:
            assert len(pool_sizes) == len(conv_channels), \
                "pool_sizes must have the same length as conv_channels"
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        current_channels = input_channels
        current_length = input_length
        
        for i, (out_channels, kernel_size, stride, pool_size) in enumerate(
            zip(conv_channels, kernel_sizes, strides, pool_sizes)
        ):
            # Add convolutional layer
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2  # Same padding
                )
            )
            
            # Calculate output size after convolution
            current_length = (current_length + 2 * (kernel_size // 2) - kernel_size) // stride + 1
            
            # Add activation
            if i < len(activations):
                if activations[i] == 'relu':
                    self.conv_layers.append(nn.ReLU())
                elif activations[i] == 'tanh':
                    self.conv_layers.append(nn.Tanh())
                elif activations[i] == 'sigmoid':
                    self.conv_layers.append(nn.Sigmoid())
                elif activations[i] == 'leaky_relu':
                    self.conv_layers.append(nn.LeakyReLU(0.1))
            
            # Add pooling layer
            if pool_size > 1:
                self.conv_layers.append(nn.MaxPool1d(pool_size))
                current_length = current_length // pool_size
            
            # Update current channels for next layer
            current_channels = out_channels
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=current_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate LSTM output size
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        current_size = lstm_output_size
        
        for i, units in enumerate(fc_layers):
            self.fc_layers.append(nn.Linear(current_size, units))
            
            # Add activation (using the remaining activations)
            activation_idx = i + len(conv_channels) + 1  # +1 for LSTM
            if activation_idx < len(activations):
                if activations[activation_idx] == 'relu':
                    self.fc_layers.append(nn.ReLU())
                elif activations[activation_idx] == 'tanh':
                    self.fc_layers.append(nn.Tanh())
                elif activations[activation_idx] == 'sigmoid':
                    self.fc_layers.append(nn.Sigmoid())
                elif activations[activation_idx] == 'leaky_relu':
                    self.fc_layers.append(nn.LeakyReLU(0.1))
            
            # Add dropout after activation (except for the last layer)
            if i < len(fc_layers) - 1 and dropout_rate > 0:
                self.fc_layers.append(nn.Dropout(dropout_rate))
            
            current_size = units
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN-LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Reshape input to (batch_size, channels, length) for 1D convolution
        x = x.permute(0, 2, 1)  # (batch_size, features, seq_len)
        
        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Reshape for LSTM: (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        for layer in self.fc_layers:
            lstm_out = layer(lstm_out)
        
        return lstm_out


class PyTorchCNNModel(BaseModel):
    """
    PyTorch CNN model implementation of the BaseModel interface.
    """
    
    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the PyTorch CNN model.
        
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
        input_length = input_shape[0]  # Window size
        
        conv_channels = model_params.get("conv_channels", [32, 64, 128])
        kernel_sizes = model_params.get("kernel_sizes", [5, 5, 5])
        strides = model_params.get("strides", [1, 1, 1])
        pool_sizes = model_params.get("pool_sizes", [2, 2, 2])
        fc_layers = model_params.get("fc_layers", [64, 32, 1])
        activations = model_params.get("activations", ["relu", "relu", "relu", "relu", "relu", "linear"])
        dropout_rate = model_params.get("dropout_rate", 0.2)
        
        # Create the model
        self.model = PyTorchCNN(
            input_channels=input_channels,
            input_length=input_length,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pool_sizes=pool_sizes,
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
        
        # Import EarlyStopping from pytorch_models
        from src.ml_models.pytorch_models import EarlyStopping
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
    def load(cls, path: str) -> 'PyTorchCNNModel':
        """
        Load a model from the given path.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            PyTorchCNNModel: Loaded model
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


class PyTorchCNNLSTMModel(BaseModel):
    """
    PyTorch CNN-LSTM hybrid model implementation of the BaseModel interface.
    """
    
    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the PyTorch CNN-LSTM model.
        
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
        input_length = input_shape[0]  # Window size
        
        conv_channels = model_params.get("conv_channels", [32, 64])
        kernel_sizes = model_params.get("kernel_sizes", [5, 5])
        strides = model_params.get("strides", [1, 1])
        pool_sizes = model_params.get("pool_sizes", [2, 2])
        lstm_hidden_size = model_params.get("lstm_hidden_size", 64)
        lstm_num_layers = model_params.get("lstm_num_layers", 2)
        fc_layers = model_params.get("fc_layers", [32, 1])
        activations = model_params.get("activations", ["relu", "relu", "relu", "linear"])
        dropout_rate = model_params.get("dropout_rate", 0.2)
        bidirectional = model_params.get("bidirectional", False)
        
        # Create the model
        self.model = PyTorchCNNLSTM(
            input_channels=input_channels,
            input_length=input_length,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pool_sizes=pool_sizes,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            fc_layers=fc_layers,
            activations=activations,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional
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
        
        # Import EarlyStopping from pytorch_models
        from src.ml_models.pytorch_models import EarlyStopping
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
    def load(cls, path: str) -> 'PyTorchCNNLSTMModel':
        """
        Load a model from the given path.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            PyTorchCNNLSTMModel: Loaded model
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
class PyTorchCNNFactory(ModelFactory):
    """Factory for creating PyTorch CNN models."""
    
    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a PyTorch CNN model."""
        return PyTorchCNNModel(input_shape, config)


class PyTorchCNNLSTMFactory(ModelFactory):
    """Factory for creating PyTorch CNN-LSTM models."""
    
    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a PyTorch CNN-LSTM model."""
        return PyTorchCNNLSTMModel(input_shape, config)


# Register factories with the model registry
ModelRegistry.register("pytorch_cnn", PyTorchCNNFactory())
ModelRegistry.register("pytorch_cnn_lstm", PyTorchCNNLSTMFactory())
# Register aliases for convenience
ModelRegistry.register("cnn", PyTorchCNNFactory())
ModelRegistry.register("cnn_lstm", PyTorchCNNLSTMFactory())