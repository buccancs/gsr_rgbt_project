# src/ml_models/pytorch_cnn_models.py

"""
PyTorch CNN model implementations for the ML pipeline.

This module contains PyTorch implementations of CNN-based models used in the pipeline,
following the BaseModel interface defined in model_interface.py.
"""

import logging
import os
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.ml_models.model_interface import BaseModel, ModelFactory, ModelRegistry
from src.ml_models.pytorch_models import EarlyStopping

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
            logging.info(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

            # Check early stopping
            if early_stopping({"val_loss": val_loss}, self.model):
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
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
            logging.info(f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

            # Check early stopping
            if early_stopping({"val_loss": val_loss}, self.model):
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
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


class PyTorchDualStreamCNNLSTM(nn.Module):
    """
    PyTorch Dual-Stream CNN-LSTM hybrid model for processing RGB and thermal video streams.

    This model has two separate CNN streams for processing RGB and thermal video frames,
    followed by a fusion layer that combines the features from both streams.
    The fused features are then passed to an LSTM to capture temporal dependencies.
    """

    def __init__(self,
                 rgb_input_shape: Tuple[int, int, int],  # (channels, height, width)
                 thermal_input_shape: Tuple[int, int, int],  # (channels, height, width)
                 cnn_filters: List[int],  # Number of filters for each CNN layer
                 cnn_kernel_sizes: List[int],  # Kernel sizes for each CNN layer
                 cnn_strides: List[int],  # Strides for each CNN layer
                 cnn_pool_sizes: List[int],  # Pool sizes for each CNN layer
                 lstm_hidden_size: int,  # Size of LSTM hidden state
                 lstm_num_layers: int,  # Number of LSTM layers
                 fc_layers: List[int],  # Sizes of fully connected layers
                 activations: List[str],  # Activation functions for each layer
                 dropout_rate: float = 0.2,  # Dropout rate
                 bidirectional: bool = False):  # Whether to use bidirectional LSTM
        """
        Initialize the Dual-Stream CNN-LSTM model.

        Args:
            rgb_input_shape (Tuple[int, int, int]): Shape of RGB input (channels, height, width)
            thermal_input_shape (Tuple[int, int, int]): Shape of thermal input (channels, height, width)
            cnn_filters (List[int]): Number of filters for each CNN layer
            cnn_kernel_sizes (List[int]): Kernel sizes for each CNN layer
            cnn_strides (List[int]): Strides for each CNN layer
            cnn_pool_sizes (List[int]): Pool sizes for each CNN layer
            lstm_hidden_size (int): Size of LSTM hidden state
            lstm_num_layers (int): Number of LSTM layers
            fc_layers (List[int]): Sizes of fully connected layers
            activations (List[str]): Activation functions for each layer
            dropout_rate (float, optional): Dropout rate
            bidirectional (bool, optional): Whether to use bidirectional LSTM
        """
        super(PyTorchDualStreamCNNLSTM, self).__init__()

        # Validate input parameters
        assert len(cnn_filters) == len(cnn_kernel_sizes) == len(cnn_strides) == len(cnn_pool_sizes), \
            "cnn_filters, cnn_kernel_sizes, cnn_strides, and cnn_pool_sizes must have the same length"

        # Build RGB CNN stream
        self.rgb_cnn_layers = self._build_cnn_stream(
            input_channels=rgb_input_shape[0],
            cnn_filters=cnn_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_strides=cnn_strides,
            cnn_pool_sizes=cnn_pool_sizes,
            activations=activations,
            dropout_rate=dropout_rate,
            stream_name="rgb"
        )

        # Build thermal CNN stream
        self.thermal_cnn_layers = self._build_cnn_stream(
            input_channels=thermal_input_shape[0],
            cnn_filters=cnn_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_strides=cnn_strides,
            cnn_pool_sizes=cnn_pool_sizes,
            activations=activations,
            dropout_rate=dropout_rate,
            stream_name="thermal"
        )

        # Calculate the output size of each CNN stream
        # This is a simplified calculation and may need to be adjusted based on the actual CNN architecture
        rgb_h, rgb_w = rgb_input_shape[1], rgb_input_shape[2]
        thermal_h, thermal_w = thermal_input_shape[1], thermal_input_shape[2]

        for kernel_size, stride, pool_size in zip(cnn_kernel_sizes, cnn_strides, cnn_pool_sizes):
            # Calculate output size after convolution
            rgb_h = ((rgb_h - kernel_size) // stride) + 1
            rgb_w = ((rgb_w - kernel_size) // stride) + 1
            thermal_h = ((thermal_h - kernel_size) // stride) + 1
            thermal_w = ((thermal_w - kernel_size) // stride) + 1

            # Calculate output size after pooling
            rgb_h = rgb_h // pool_size
            rgb_w = rgb_w // pool_size
            thermal_h = thermal_h // pool_size
            thermal_w = thermal_w // pool_size

        # Final CNN output size for each stream
        rgb_cnn_output_size = cnn_filters[-1] * rgb_h * rgb_w
        thermal_cnn_output_size = cnn_filters[-1] * thermal_h * thermal_w

        # Fusion layer to combine features from both streams
        self.fusion_layer = nn.Linear(rgb_cnn_output_size + thermal_cnn_output_size, cnn_filters[-1])

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
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

            # Add activation
            activation_idx = i + len(cnn_filters) * 2 + 1  # *2 for two streams, +1 for LSTM
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

    def _build_cnn_stream(self, input_channels: int, cnn_filters: List[int],
                          cnn_kernel_sizes: List[int], cnn_strides: List[int],
                          cnn_pool_sizes: List[int], activations: List[str],
                          dropout_rate: float, stream_name: str) -> nn.ModuleList:
        """
        Build a CNN stream for processing video frames.

        Args:
            input_channels (int): Number of input channels
            cnn_filters (List[int]): Number of filters for each CNN layer
            cnn_kernel_sizes (List[int]): Kernel sizes for each CNN layer
            cnn_strides (List[int]): Strides for each CNN layer
            cnn_pool_sizes (List[int]): Pool sizes for each CNN layer
            activations (List[str]): Activation functions for each layer
            dropout_rate (float): Dropout rate
            stream_name (str): Name of the stream (for logging)

        Returns:
            nn.ModuleList: List of CNN layers
        """
        cnn_layers = nn.ModuleList()
        current_channels = input_channels

        for i, (filters, kernel_size, stride, pool_size) in enumerate(
                zip(cnn_filters, cnn_kernel_sizes, cnn_strides, cnn_pool_sizes)
        ):
            # Add convolutional layer
            cnn_layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0  # No padding
                )
            )

            # Add activation
            if i < len(activations):
                if activations[i] == 'relu':
                    cnn_layers.append(nn.ReLU())
                elif activations[i] == 'tanh':
                    cnn_layers.append(nn.Tanh())
                elif activations[i] == 'sigmoid':
                    cnn_layers.append(nn.Sigmoid())
                elif activations[i] == 'leaky_relu':
                    cnn_layers.append(nn.LeakyReLU(0.1))

            # Add pooling layer
            if pool_size > 1:
                cnn_layers.append(nn.MaxPool2d(pool_size))

            # Add dropout (except for the last layer)
            if i < len(cnn_filters) - 1 and dropout_rate > 0:
                cnn_layers.append(nn.Dropout(dropout_rate))

            # Update current channels for next layer
            current_channels = filters

        return cnn_layers

    def forward(self, rgb_frames: torch.Tensor, thermal_frames: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Dual-Stream CNN-LSTM model.

        Args:
            rgb_frames (torch.Tensor): RGB video frames of shape (batch_size, seq_len, channels, height, width)
            thermal_frames (torch.Tensor): Thermal video frames of shape (batch_size, seq_len, channels, height, width)

        Returns:
            torch.Tensor: Output tensor
        """
        batch_size, seq_len = rgb_frames.shape[0], rgb_frames.shape[1]

        # Process each frame in the sequence
        fused_features = []

        for t in range(seq_len):
            # Get current frames
            rgb_frame = rgb_frames[:, t]  # (batch_size, channels, height, width)
            thermal_frame = thermal_frames[:, t]  # (batch_size, channels, height, width)

            # Process RGB stream
            rgb_features = rgb_frame
            for layer in self.rgb_cnn_layers:
                rgb_features = layer(rgb_features)

            # Process thermal stream
            thermal_features = thermal_frame
            for layer in self.thermal_cnn_layers:
                thermal_features = layer(thermal_features)

            # Flatten features
            rgb_features = rgb_features.view(batch_size, -1)
            thermal_features = thermal_features.view(batch_size, -1)

            # Concatenate features from both streams
            combined_features = torch.cat([rgb_features, thermal_features], dim=1)

            # Fuse features
            fused = self.fusion_layer(combined_features)
            fused_features.append(fused)

        # Stack features from all frames
        sequence = torch.stack(fused_features, dim=1)  # (batch_size, seq_len, fused_dim)

        # Pass through LSTM
        lstm_out, _ = self.lstm(sequence)

        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]

        # Pass through fully connected layers
        for layer in self.fc_layers:
            lstm_out = layer(lstm_out)

        return lstm_out


class PyTorchDualStreamCNNLSTMModel(BaseModel):
    """
    PyTorch Dual-Stream CNN-LSTM model implementation of the BaseModel interface.

    This model processes both RGB and thermal video streams to predict GSR signals.
    """

    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the PyTorch Dual-Stream CNN-LSTM model.

        Args:
            input_shape (Tuple[int, int]): Shape of the input data (window_size, features)
                                          This is a placeholder and will be overridden by the
                                          rgb_input_shape and thermal_input_shape from config
            config (Dict[str, Any]): Model configuration parameters
        """
        self.input_shape = input_shape
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract model parameters
        model_params = config.get("model_params", {})

        # Get input shapes for RGB and thermal streams
        rgb_input_shape = model_params.get("rgb_input_shape", (3, 64, 64))  # (channels, height, width)
        thermal_input_shape = model_params.get("thermal_input_shape", (3, 64, 64))  # (channels, height, width)

        # CNN parameters
        cnn_filters = model_params.get("cnn_filters", [32, 64, 128])
        cnn_kernel_sizes = model_params.get("cnn_kernel_sizes", [3, 3, 3])
        cnn_strides = model_params.get("cnn_strides", [1, 1, 1])
        cnn_pool_sizes = model_params.get("cnn_pool_sizes", [2, 2, 2])

        # LSTM parameters
        lstm_hidden_size = model_params.get("lstm_hidden_size", 128)
        lstm_num_layers = model_params.get("lstm_num_layers", 2)
        bidirectional = model_params.get("bidirectional", False)

        # FC parameters
        fc_layers = model_params.get("fc_layers", [64, 32, 1])

        # Other parameters
        activations = model_params.get("activations", ["relu"] * (len(cnn_filters) * 2 + len(fc_layers)))
        dropout_rate = model_params.get("dropout_rate", 0.2)

        # Create the model
        self.model = PyTorchDualStreamCNNLSTM(
            rgb_input_shape=rgb_input_shape,
            thermal_input_shape=thermal_input_shape,
            cnn_filters=cnn_filters,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_strides=cnn_strides,
            cnn_pool_sizes=cnn_pool_sizes,
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
            momentum = optimizer_params.get("momentum", 0.9)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Set up loss function
        loss_params = config.get("loss_params", {})
        loss_type = loss_params.get("type", "mse").lower()

        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            delta = loss_params.get("delta", 1.0)
            self.criterion = nn.SmoothL1Loss(beta=delta)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Initialize training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_metrics": [],
            "val_metrics": []
        }

        logging.info(f"Initialized PyTorchDualStreamCNNLSTMModel on device: {self.device}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the given data.

        Args:
            X (np.ndarray): Input features, expected to be a tuple of (rgb_frames, thermal_frames)
                           where each element is of shape (n_samples, seq_len, channels, height, width)
            y (np.ndarray): Target values of shape (n_samples,)
            **kwargs: Additional arguments for training
                - batch_size (int): Batch size for training
                - epochs (int): Number of epochs to train
                - validation_split (float): Fraction of data to use for validation
                - validation_data (tuple): Tuple of (X_val, y_val) for validation
                - early_stopping (bool): Whether to use early stopping
                - patience (int): Patience for early stopping
                - min_delta (float): Minimum change to qualify as improvement
                - verbose (bool): Whether to print progress

        Returns:
            Dict[str, Any]: Training history
        """
        # Extract training parameters
        batch_size = kwargs.get("batch_size", 32)
        epochs = kwargs.get("epochs", 100)
        validation_split = kwargs.get("validation_split", 0.2)
        validation_data = kwargs.get("validation_data", None)
        early_stopping = kwargs.get("early_stopping", True)
        patience = kwargs.get("patience", 10)
        min_delta = kwargs.get("min_delta", 0.001)
        verbose = kwargs.get("verbose", True)

        # Unpack input data
        if not isinstance(X, tuple) or len(X) != 2:
            raise ValueError("Input X must be a tuple of (rgb_frames, thermal_frames)")

        rgb_frames, thermal_frames = X

        # Convert numpy arrays to PyTorch tensors
        rgb_tensor = torch.tensor(rgb_frames, dtype=torch.float32)
        thermal_tensor = torch.tensor(thermal_frames, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        # Create dataset
        dataset = TensorDataset(rgb_tensor, thermal_tensor, y_tensor)

        # Split dataset into training and validation
        if validation_data is None:
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        else:
            # Unpack validation data
            X_val, y_val = validation_data
            if not isinstance(X_val, tuple) or len(X_val) != 2:
                raise ValueError("Validation input X_val must be a tuple of (rgb_frames, thermal_frames)")

            rgb_val, thermal_val = X_val

            # Convert validation data to tensors
            rgb_val_tensor = torch.tensor(rgb_val, dtype=torch.float32)
            thermal_val_tensor = torch.tensor(thermal_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

            # Create datasets
            train_dataset = dataset
            val_dataset = TensorDataset(rgb_val_tensor, thermal_val_tensor, y_val_tensor)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize early stopping
        if early_stopping:
            early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for rgb_batch, thermal_batch, y_batch in train_loader:
                # Move data to device
                rgb_batch = rgb_batch.to(self.device)
                thermal_batch = thermal_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(rgb_batch, thermal_batch)

                # Calculate loss
                loss = self.criterion(outputs, y_batch)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Accumulate loss
                train_loss += loss.item() * rgb_batch.size(0)

            # Calculate average training loss
            train_loss /= len(train_loader.dataset)

            # Validation phase
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for rgb_batch, thermal_batch, y_batch in val_loader:
                    # Move data to device
                    rgb_batch = rgb_batch.to(self.device)
                    thermal_batch = thermal_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    outputs = self.model(rgb_batch, thermal_batch)

                    # Calculate loss
                    loss = self.criterion(outputs, y_batch)

                    # Accumulate loss
                    val_loss += loss.item() * rgb_batch.size(0)

            # Calculate average validation loss
            val_loss /= len(val_loader.dataset)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Check for early stopping
            if early_stopping:
                should_stop = early_stopper({"val_loss": val_loss}, self.model)
                if should_stop:
                    logging.info(f"Early stopping triggered at epoch {epoch + 1}")
                    # Restore best weights
                    early_stopper.restore_best_weights(self.model)
                    break

        return self.history

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the model.

        Args:
            X (np.ndarray): Input features, expected to be a tuple of (rgb_frames, thermal_frames)
                           where each element is of shape (n_samples, seq_len, channels, height, width)
            **kwargs: Additional arguments for prediction

        Returns:
            np.ndarray: Predictions
        """
        # Unpack input data
        if not isinstance(X, tuple) or len(X) != 2:
            raise ValueError("Input X must be a tuple of (rgb_frames, thermal_frames)")

        rgb_frames, thermal_frames = X

        # Convert numpy arrays to PyTorch tensors
        rgb_tensor = torch.tensor(rgb_frames, dtype=torch.float32).to(self.device)
        thermal_tensor = torch.tensor(thermal_frames, dtype=torch.float32).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            predictions = self.model(rgb_tensor, thermal_tensor).cpu().numpy()

        return predictions.flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on the given data.

        Args:
            X (np.ndarray): Input features, expected to be a tuple of (rgb_frames, thermal_frames)
                           where each element is of shape (n_samples, seq_len, channels, height, width)
            y (np.ndarray): Target values of shape (n_samples,)
            **kwargs: Additional arguments for evaluation

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Unpack input data
        if not isinstance(X, tuple) or len(X) != 2:
            raise ValueError("Input X must be a tuple of (rgb_frames, thermal_frames)")

        rgb_frames, thermal_frames = X

        # Convert numpy arrays to PyTorch tensors
        rgb_tensor = torch.tensor(rgb_frames, dtype=torch.float32).to(self.device)
        thermal_tensor = torch.tensor(thermal_frames, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            predictions = self.model(rgb_tensor, thermal_tensor)
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
    def load(cls, path: str) -> 'PyTorchDualStreamCNNLSTMModel':
        """
        Load a model from the given path.

        Args:
            path (str): Path to load the model from

        Returns:
            PyTorchDualStreamCNNLSTMModel: Loaded model
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


class PyTorchDualStreamCNNLSTMFactory(ModelFactory):
    """Factory for creating PyTorch Dual-Stream CNN-LSTM models."""

    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a PyTorch Dual-Stream CNN-LSTM model."""
        return PyTorchDualStreamCNNLSTMModel(input_shape, config)


# Register factories with the model registry
ModelRegistry.register("pytorch_cnn", PyTorchCNNFactory())
ModelRegistry.register("pytorch_cnn_lstm", PyTorchCNNLSTMFactory())
ModelRegistry.register("pytorch_dual_stream_cnn_lstm", PyTorchDualStreamCNNLSTMFactory())
# Register aliases for convenience
ModelRegistry.register("cnn", PyTorchCNNFactory())
ModelRegistry.register("cnn_lstm", PyTorchCNNLSTMFactory())
ModelRegistry.register("dual_stream_cnn_lstm", PyTorchDualStreamCNNLSTMFactory())
