# src/ml_models/tensorflow_cnn_models.py

"""
TensorFlow CNN model implementations for time series regression.

This module contains TensorFlow implementations of CNN models for time series data,
including basic CNN, CNN-LSTM, and dual-stream CNN-LSTM architectures.
"""

import logging
import os
from typing import Dict, Any, Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Sequential, Model

from src.ml_models.model_interface import BaseModel, ModelFactory, ModelRegistry

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


class TensorFlowCNN(Model):
    """
    TensorFlow CNN model for time series regression.

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
        super(TensorFlowCNN, self).__init__()

        # Validate input parameters
        assert len(conv_channels) == len(kernel_sizes) == len(strides), \
            "conv_channels, kernel_sizes, and strides must have the same length"

        if pool_sizes is None:
            pool_sizes = [2] * len(conv_channels)
        else:
            assert len(pool_sizes) == len(conv_channels), \
                "pool_sizes must have the same length as conv_channels"

        # Build the model
        self.model_layers = []

        # Build convolutional layers
        for i, (filters, kernel_size, stride, pool_size) in enumerate(
                zip(conv_channels, kernel_sizes, strides, pool_sizes)
        ):
            # Add convolutional layer
            self.model_layers.append(
                layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding='same'
                )
            )

            # Add activation
            if i < len(activations):
                if activations[i] == 'relu':
                    self.model_layers.append(layers.ReLU())
                elif activations[i] == 'tanh':
                    self.model_layers.append(layers.Activation('tanh'))
                elif activations[i] == 'sigmoid':
                    self.model_layers.append(layers.Activation('sigmoid'))
                elif activations[i] == 'leaky_relu':
                    self.model_layers.append(layers.LeakyReLU(0.1))

            # Add pooling layer
            if pool_size > 1:
                self.model_layers.append(
                    layers.MaxPool1D(pool_size=pool_size)
                )

        # Flatten layer
        self.flatten_layer = layers.Flatten()

        # Build fully connected layers
        self.fc_layers_list = []
        for i, units in enumerate(fc_layers):
            self.fc_layers_list.append(layers.Dense(units))

            # Add activation (using the remaining activations)
            activation_idx = i + len(conv_channels)
            if activation_idx < len(activations):
                if activations[activation_idx] == 'relu':
                    self.fc_layers_list.append(layers.ReLU())
                elif activations[activation_idx] == 'tanh':
                    self.fc_layers_list.append(layers.Activation('tanh'))
                elif activations[activation_idx] == 'sigmoid':
                    self.fc_layers_list.append(layers.Activation('sigmoid'))
                elif activations[activation_idx] == 'leaky_relu':
                    self.fc_layers_list.append(layers.LeakyReLU(0.1))
                elif activations[activation_idx] == 'linear':
                    # Linear activation is the default in TensorFlow
                    pass

            # Add dropout after activation (except for the last layer)
            if i < len(fc_layers) - 1 and dropout_rate > 0:
                self.fc_layers_list.append(layers.Dropout(dropout_rate))

    def call(self, inputs, training=None):
        """
        Forward pass of the CNN model.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features)
            training: Boolean indicating whether the model is in training mode

        Returns:
            Tensor: Output tensor
        """
        # Input is already in the correct format for 1D convolution: (batch_size, seq_len, features)
        x = inputs

        # Pass through convolutional layers
        for layer in self.model_layers:
            x = layer(x)

        # Flatten the output
        x = self.flatten_layer(x)

        # Pass through fully connected layers
        for layer in self.fc_layers_list:
            x = layer(x)

        return x


class TensorFlowCNNLSTM(Model):
    """
    TensorFlow CNN-LSTM hybrid model for time series regression.

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
        super(TensorFlowCNNLSTM, self).__init__()

        # Validate input parameters
        assert len(conv_channels) == len(kernel_sizes) == len(strides), \
            "conv_channels, kernel_sizes, and strides must have the same length"

        if pool_sizes is None:
            pool_sizes = [2] * len(conv_channels)
        else:
            assert len(pool_sizes) == len(conv_channels), \
                "pool_sizes must have the same length as conv_channels"

        # Build the model
        self.model_layers = []

        # Build convolutional layers
        for i, (filters, kernel_size, stride, pool_size) in enumerate(
                zip(conv_channels, kernel_sizes, strides, pool_sizes)
        ):
            # Add convolutional layer
            self.model_layers.append(
                layers.Conv1D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding='same'
                )
            )

            # Add activation
            if i < len(activations):
                if activations[i] == 'relu':
                    self.model_layers.append(layers.ReLU())
                elif activations[i] == 'tanh':
                    self.model_layers.append(layers.Activation('tanh'))
                elif activations[i] == 'sigmoid':
                    self.model_layers.append(layers.Activation('sigmoid'))
                elif activations[i] == 'leaky_relu':
                    self.model_layers.append(layers.LeakyReLU(0.1))

            # Add pooling layer
            if pool_size > 1:
                self.model_layers.append(
                    layers.MaxPool1D(pool_size=pool_size)
                )

        # LSTM layers
        self.lstm_layers = []
        for i in range(lstm_num_layers):
            return_sequences = i < lstm_num_layers - 1  # Only last layer doesn't return sequences

            if bidirectional:
                self.lstm_layers.append(
                    layers.Bidirectional(
                        layers.LSTM(
                            units=lstm_hidden_size,
                            return_sequences=return_sequences,
                            dropout=dropout_rate if i < lstm_num_layers - 1 else 0
                        )
                    )
                )
            else:
                self.lstm_layers.append(
                    layers.LSTM(
                        units=lstm_hidden_size,
                        return_sequences=return_sequences,
                        dropout=dropout_rate if i < lstm_num_layers - 1 else 0
                    )
                )

        # Fully connected layers
        self.fc_layers_list = []
        for i, units in enumerate(fc_layers):
            self.fc_layers_list.append(layers.Dense(units))

            # Add activation (using the remaining activations)
            activation_idx = i + len(conv_channels) + 1  # +1 for LSTM
            if activation_idx < len(activations):
                if activations[activation_idx] == 'relu':
                    self.fc_layers_list.append(layers.ReLU())
                elif activations[activation_idx] == 'tanh':
                    self.fc_layers_list.append(layers.Activation('tanh'))
                elif activations[activation_idx] == 'sigmoid':
                    self.fc_layers_list.append(layers.Activation('sigmoid'))
                elif activations[activation_idx] == 'leaky_relu':
                    self.fc_layers_list.append(layers.LeakyReLU(0.1))
                elif activations[activation_idx] == 'linear':
                    # Linear activation is the default in TensorFlow
                    pass

            # Add dropout after activation (except for the last layer)
            if i < len(fc_layers) - 1 and dropout_rate > 0:
                self.fc_layers_list.append(layers.Dropout(dropout_rate))

    def call(self, inputs, training=None):
        """
        Forward pass of the CNN-LSTM model.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features)
            training: Boolean indicating whether the model is in training mode

        Returns:
            Tensor: Output tensor
        """
        # Input is already in the correct format for 1D convolution: (batch_size, seq_len, features)
        x = inputs

        # Pass through convolutional layers
        for layer in self.model_layers:
            x = layer(x)

        # Pass through LSTM layers (x is already in the correct format)
        for layer in self.lstm_layers:
            x = layer(x)

        # Pass through fully connected layers
        for layer in self.fc_layers_list:
            x = layer(x)

        return x


class TensorFlowCNNModel(BaseModel):
    """
    TensorFlow CNN model implementation of the BaseModel interface.
    """

    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the TensorFlow CNN model.

        Args:
            input_shape (Tuple[int, int]): Shape of the input data (window_size, features)
            config (Dict[str, Any]): Model configuration parameters
        """
        self.input_shape = input_shape
        self.config = config

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
        self.model = TensorFlowCNN(
            input_channels=input_channels,
            input_length=input_length,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            pool_sizes=pool_sizes,
            fc_layers=fc_layers,
            activations=activations,
            dropout_rate=dropout_rate
        )

        # Set up optimizer
        optimizer_params = config.get("optimizer_params", {})
        optimizer_type = optimizer_params.get("type", "adam").lower()
        lr = optimizer_params.get("lr", 0.001)
        weight_decay = optimizer_params.get("weight_decay", 1e-5)

        if optimizer_type == "adam":
            self.optimizer = optimizers.Adam(
                learning_rate=lr
            )
        elif optimizer_type == "sgd":
            self.optimizer = optimizers.SGD(
                learning_rate=lr
            )
        else:
            # Default to Adam
            self.optimizer = optimizers.Adam(
                learning_rate=lr
            )

        # Set up loss function
        loss_fn = config.get("loss_fn", "mse").lower()
        if loss_fn == "mse":
            self.loss = tf.keras.losses.MeanSquaredError()
        elif loss_fn == "mae" or loss_fn == "mean_absolute_error":
            self.loss = tf.keras.losses.MeanAbsoluteError()
        else:
            # Default to MSE
            self.loss = tf.keras.losses.MeanSquaredError()

        # Compile the model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['mse', 'mae']
        )

        # Training history
        self.history = None

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
        # Get training parameters
        train_params = self.config.get("train_params", {})
        batch_size = train_params.get("batch_size", 32)
        epochs = train_params.get("epochs", 100)
        validation_split = train_params.get("validation_split", 0.2)

        # Set up early stopping
        early_stopping_params = train_params.get("early_stopping", {})
        patience = early_stopping_params.get("patience", 10)
        monitor = early_stopping_params.get("monitor", "val_loss")

        callbacks_list = [
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True
            )
        ]

        # Train the model
        self.history = self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1
        )

        return self.history.history

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the model.

        Args:
            X (np.ndarray): Input features
            **kwargs: Additional arguments for prediction

        Returns:
            np.ndarray: Predictions
        """
        predictions = self.model.predict(X)
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
        # Evaluate the model
        metrics = self.model.evaluate(X, y, verbose=0)

        # Create a dictionary of metrics
        metrics_dict = {
            "loss": metrics[0],
            "mse": metrics[1],
            "mae": metrics[2]
        }

        return metrics_dict

    def save(self, path: str) -> None:
        """
        Save the model to the given path.

        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model
        self.model.save(path)

        logging.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'TensorFlowCNNModel':
        """
        Load a model from the given path.

        Args:
            path (str): Path to load the model from

        Returns:
            TensorFlowCNNModel: Loaded model
        """
        # Load the model
        model = tf.keras.models.load_model(path)

        # Create a dummy instance (this is a bit of a hack, but necessary for the interface)
        # In a real implementation, you would also save and load the configuration
        dummy_input_shape = (model.input_shape[1], model.input_shape[2])
        dummy_config = {}

        instance = cls(dummy_input_shape, dummy_config)
        instance.model = model

        logging.info(f"Model loaded from {path}")
        return instance


class TensorFlowCNNLSTMModel(BaseModel):
    """
    TensorFlow CNN-LSTM model implementation of the BaseModel interface.
    """

    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the TensorFlow CNN-LSTM model.

        Args:
            input_shape (Tuple[int, int]): Shape of the input data (window_size, features)
            config (Dict[str, Any]): Model configuration parameters
        """
        self.input_shape = input_shape
        self.config = config

        # Extract model parameters
        model_params = config.get("model_params", {})
        input_channels = input_shape[1]  # Number of features
        input_length = input_shape[0]  # Window size

        conv_channels = model_params.get("conv_channels", [32, 64, 128])
        kernel_sizes = model_params.get("kernel_sizes", [5, 5, 5])
        strides = model_params.get("strides", [1, 1, 1])
        pool_sizes = model_params.get("pool_sizes", [2, 2, 2])
        lstm_hidden_size = model_params.get("lstm_hidden_size", 64)
        lstm_num_layers = model_params.get("lstm_num_layers", 1)
        fc_layers = model_params.get("fc_layers", [32, 1])
        activations = model_params.get("activations", ["relu", "relu", "relu", "tanh", "relu", "linear"])
        dropout_rate = model_params.get("dropout_rate", 0.2)
        bidirectional = model_params.get("bidirectional", False)

        # Create the model
        self.model = TensorFlowCNNLSTM(
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
        )

        # Set up optimizer
        optimizer_params = config.get("optimizer_params", {})
        optimizer_type = optimizer_params.get("type", "adam").lower()
        lr = optimizer_params.get("lr", 0.001)
        weight_decay = optimizer_params.get("weight_decay", 1e-5)

        if optimizer_type == "adam":
            self.optimizer = optimizers.Adam(
                learning_rate=lr
            )
        elif optimizer_type == "sgd":
            self.optimizer = optimizers.SGD(
                learning_rate=lr
            )
        else:
            # Default to Adam
            self.optimizer = optimizers.Adam(
                learning_rate=lr
            )

        # Set up loss function
        loss_fn = config.get("loss_fn", "mse").lower()
        if loss_fn == "mse":
            self.loss = tf.keras.losses.MeanSquaredError()
        elif loss_fn == "mae" or loss_fn == "mean_absolute_error":
            self.loss = tf.keras.losses.MeanAbsoluteError()
        else:
            # Default to MSE
            self.loss = tf.keras.losses.MeanSquaredError()

        # Compile the model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['mse', 'mae']
        )

        # Training history
        self.history = None

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
        # Get training parameters
        train_params = self.config.get("train_params", {})
        batch_size = train_params.get("batch_size", 32)
        epochs = train_params.get("epochs", 100)
        validation_split = train_params.get("validation_split", 0.2)

        # Set up early stopping
        early_stopping_params = train_params.get("early_stopping", {})
        patience = early_stopping_params.get("patience", 10)
        monitor = early_stopping_params.get("monitor", "val_loss")

        callbacks_list = [
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True
            )
        ]

        # Train the model
        self.history = self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1
        )

        return self.history.history

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the model.

        Args:
            X (np.ndarray): Input features
            **kwargs: Additional arguments for prediction

        Returns:
            np.ndarray: Predictions
        """
        predictions = self.model.predict(X)
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
        # Evaluate the model
        metrics = self.model.evaluate(X, y, verbose=0)

        # Create a dictionary of metrics
        metrics_dict = {
            "loss": metrics[0],
            "mse": metrics[1],
            "mae": metrics[2]
        }

        return metrics_dict

    def save(self, path: str) -> None:
        """
        Save the model to the given path.

        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model
        self.model.save(path)

        logging.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'TensorFlowCNNLSTMModel':
        """
        Load a model from the given path.

        Args:
            path (str): Path to load the model from

        Returns:
            TensorFlowCNNLSTMModel: Loaded model
        """
        # Load the model
        model = tf.keras.models.load_model(path)

        # Create a dummy instance (this is a bit of a hack, but necessary for the interface)
        # In a real implementation, you would also save and load the configuration
        dummy_input_shape = (model.input_shape[1], model.input_shape[2])
        dummy_config = {}

        instance = cls(dummy_input_shape, dummy_config)
        instance.model = model

        logging.info(f"Model loaded from {path}")
        return instance


class TensorFlowCNNFactory(ModelFactory):
    """Factory for creating TensorFlow CNN models."""

    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a TensorFlow CNN model."""
        return TensorFlowCNNModel(input_shape, config)


class TensorFlowCNNLSTMFactory(ModelFactory):
    """Factory for creating TensorFlow CNN-LSTM models."""

    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a TensorFlow CNN-LSTM model."""
        return TensorFlowCNNLSTMModel(input_shape, config)


class TensorFlowDualStreamCNNLSTM(Model):
    """
    TensorFlow Dual-Stream CNN-LSTM hybrid model for processing RGB and thermal video streams.

    This model has two separate CNN streams for processing RGB and thermal video frames,
    followed by a fusion layer that combines the features from both streams.
    The fused features are then passed to an LSTM to capture temporal dependencies.
    """

    def __init__(self,
                 rgb_input_shape: Tuple[int, int, int],  # (height, width, channels)
                 thermal_input_shape: Tuple[int, int, int],  # (height, width, channels)
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
            rgb_input_shape (Tuple[int, int, int]): Shape of RGB input (height, width, channels)
            thermal_input_shape (Tuple[int, int, int]): Shape of thermal input (height, width, channels)
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
        super(TensorFlowDualStreamCNNLSTM, self).__init__()

        # Validate input parameters
        assert len(cnn_filters) == len(cnn_kernel_sizes) == len(cnn_strides) == len(cnn_pool_sizes), \
            "cnn_filters, cnn_kernel_sizes, cnn_strides, and cnn_pool_sizes must have the same length"

        # Build RGB CNN stream
        self.rgb_cnn_layers = self._build_cnn_stream(
            input_channels=rgb_input_shape[2],
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
            input_channels=thermal_input_shape[2],
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
        rgb_h, rgb_w = rgb_input_shape[0], rgb_input_shape[1]
        thermal_h, thermal_w = thermal_input_shape[0], thermal_input_shape[1]

        for kernel_size, stride, pool_size in zip(cnn_kernel_sizes, cnn_strides, cnn_pool_sizes):
            # Calculate output size after convolution (with 'valid' padding)
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
        self.fusion_layer = layers.Dense(cnn_filters[-1])

        # LSTM layers
        self.lstm_layers = []
        for i in range(lstm_num_layers):
            return_sequences = i < lstm_num_layers - 1  # Only last layer doesn't return sequences

            if bidirectional:
                self.lstm_layers.append(
                    layers.Bidirectional(
                        layers.LSTM(
                            units=lstm_hidden_size,
                            return_sequences=return_sequences,
                            dropout=dropout_rate if i < lstm_num_layers - 1 else 0
                        )
                    )
                )
            else:
                self.lstm_layers.append(
                    layers.LSTM(
                        units=lstm_hidden_size,
                        return_sequences=return_sequences,
                        dropout=dropout_rate if i < lstm_num_layers - 1 else 0
                    )
                )

        # Build fully connected layers
        self.fc_layers_list = []
        for i, units in enumerate(fc_layers):
            self.fc_layers_list.append(layers.Dense(units))

            # Add activation
            activation_idx = i + len(cnn_filters) * 2 + 1  # *2 for two streams, +1 for LSTM
            if activation_idx < len(activations):
                if activations[activation_idx] == 'relu':
                    self.fc_layers_list.append(layers.ReLU())
                elif activations[activation_idx] == 'tanh':
                    self.fc_layers_list.append(layers.Activation('tanh'))
                elif activations[activation_idx] == 'sigmoid':
                    self.fc_layers_list.append(layers.Activation('sigmoid'))
                elif activations[activation_idx] == 'leaky_relu':
                    self.fc_layers_list.append(layers.LeakyReLU(0.1))
                elif activations[activation_idx] == 'linear':
                    # Linear activation is the default in TensorFlow
                    pass

            # Add dropout after activation (except for the last layer)
            if i < len(fc_layers) - 1 and dropout_rate > 0:
                self.fc_layers_list.append(layers.Dropout(dropout_rate))

    def _build_cnn_stream(self, input_channels: int, cnn_filters: List[int],
                          cnn_kernel_sizes: List[int], cnn_strides: List[int],
                          cnn_pool_sizes: List[int], activations: List[str],
                          dropout_rate: float, stream_name: str) -> List:
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
            List: List of CNN layers
        """
        cnn_layers = []
        current_channels = input_channels

        for i, (filters, kernel_size, stride, pool_size) in enumerate(
                zip(cnn_filters, cnn_kernel_sizes, cnn_strides, cnn_pool_sizes)
        ):
            # Add convolutional layer
            cnn_layers.append(
                layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding='valid'  # No padding (equivalent to PyTorch's padding=0)
                )
            )

            # Add activation
            if i < len(activations):
                if activations[i] == 'relu':
                    cnn_layers.append(layers.ReLU())
                elif activations[i] == 'tanh':
                    cnn_layers.append(layers.Activation('tanh'))
                elif activations[i] == 'sigmoid':
                    cnn_layers.append(layers.Activation('sigmoid'))
                elif activations[i] == 'leaky_relu':
                    cnn_layers.append(layers.LeakyReLU(0.1))

            # Add pooling layer
            if pool_size > 1:
                cnn_layers.append(layers.MaxPool2D(pool_size=pool_size))

            # Add dropout (except for the last layer)
            if i < len(cnn_filters) - 1 and dropout_rate > 0:
                cnn_layers.append(layers.Dropout(dropout_rate))

            # Update current channels for next layer
            current_channels = filters

        return cnn_layers

    def call(self, inputs, training=None):
        """
        Forward pass of the Dual-Stream CNN-LSTM model.

        Args:
            inputs: Tuple of (rgb_frames, thermal_frames)
                   rgb_frames: RGB video frames of shape (batch_size, seq_len, height, width, channels)
                   thermal_frames: Thermal video frames of shape (batch_size, seq_len, height, width, channels)
            training: Boolean indicating whether the model is in training mode

        Returns:
            Tensor: Output tensor
        """
        rgb_frames, thermal_frames = inputs
        batch_size, seq_len = tf.shape(rgb_frames)[0], tf.shape(rgb_frames)[1]

        # Process each frame in the sequence
        fused_features = []

        for t in range(seq_len):
            # Get current frames
            rgb_frame = rgb_frames[:, t]  # (batch_size, height, width, channels)
            thermal_frame = thermal_frames[:, t]  # (batch_size, height, width, channels)

            # Process RGB stream
            rgb_features = rgb_frame
            for layer in self.rgb_cnn_layers:
                rgb_features = layer(rgb_features)

            # Process thermal stream
            thermal_features = thermal_frame
            for layer in self.thermal_cnn_layers:
                thermal_features = layer(thermal_features)

            # Flatten features
            rgb_features = layers.Flatten()(rgb_features)
            thermal_features = layers.Flatten()(thermal_features)

            # Concatenate features from both streams
            combined_features = layers.Concatenate()([rgb_features, thermal_features])

            # Fuse features
            fused = self.fusion_layer(combined_features)
            fused_features.append(fused)

        # Stack features from all frames
        sequence = tf.stack(fused_features, axis=1)  # (batch_size, seq_len, fused_dim)

        # Pass through LSTM layers
        x = sequence
        for layer in self.lstm_layers:
            x = layer(x)

        # Pass through fully connected layers
        for layer in self.fc_layers_list:
            x = layer(x)

        return x


class TensorFlowDualStreamCNNLSTMModel(BaseModel):
    """
    TensorFlow Dual-Stream CNN-LSTM model implementation of the BaseModel interface.

    This model processes both RGB and thermal video streams to predict GSR signals.
    """

    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the TensorFlow Dual-Stream CNN-LSTM model.

        Args:
            input_shape (Tuple[int, int]): Shape of the input data (window_size, features)
                                          This is a placeholder and will be overridden by the
                                          rgb_input_shape and thermal_input_shape from config
            config (Dict[str, Any]): Model configuration parameters
        """
        self.input_shape = input_shape
        self.config = config

        # Extract model parameters
        model_params = config.get("model_params", {})

        # Get input shapes for RGB and thermal streams
        rgb_input_shape = model_params.get("rgb_input_shape", (64, 64, 3))  # (height, width, channels)
        thermal_input_shape = model_params.get("thermal_input_shape", (64, 64, 3))  # (height, width, channels)

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
        self.model = TensorFlowDualStreamCNNLSTM(
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
        )

        # Set up optimizer
        optimizer_params = config.get("optimizer_params", {})
        optimizer_type = optimizer_params.get("type", "adam").lower()
        lr = optimizer_params.get("lr", 0.001)
        weight_decay = optimizer_params.get("weight_decay", 1e-5)

        if optimizer_type == "adam":
            self.optimizer = optimizers.Adam(
                learning_rate=lr
            )
        elif optimizer_type == "sgd":
            self.optimizer = optimizers.SGD(
                learning_rate=lr
            )
        else:
            # Default to Adam
            self.optimizer = optimizers.Adam(
                learning_rate=lr
            )

        # Set up loss function
        loss_fn = config.get("loss_fn", "mse").lower()
        if loss_fn == "mse":
            self.loss = tf.keras.losses.MeanSquaredError()
        elif loss_fn == "mae" or loss_fn == "mean_absolute_error":
            self.loss = tf.keras.losses.MeanAbsoluteError()
        else:
            # Default to MSE
            self.loss = tf.keras.losses.MeanSquaredError()

        # Compile the model
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['mse', 'mae']
        )

        # Training history
        self.history = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the given data.

        Args:
            X (np.ndarray): Input features - should be a tuple of (rgb_frames, thermal_frames)
                           Each with shape (samples, seq_len, height, width, channels)
            y (np.ndarray): Target values of shape (samples,)
            **kwargs: Additional arguments for training

        Returns:
            Dict[str, Any]: Training history
        """
        # Get training parameters
        train_params = self.config.get("train_params", {})
        batch_size = train_params.get("batch_size", 32)
        epochs = train_params.get("epochs", 100)
        validation_split = train_params.get("validation_split", 0.2)

        # Set up early stopping
        early_stopping_params = train_params.get("early_stopping", {})
        patience = early_stopping_params.get("patience", 10)
        monitor = early_stopping_params.get("monitor", "val_loss")

        callbacks_list = [
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True
            )
        ]

        # Train the model
        self.history = self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1
        )

        return self.history.history

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        Make predictions with the model.

        Args:
            X (np.ndarray): Input features - should be a tuple of (rgb_frames, thermal_frames)
            **kwargs: Additional arguments for prediction

        Returns:
            np.ndarray: Predictions
        """
        predictions = self.model.predict(X)
        return predictions.flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on the given data.

        Args:
            X (np.ndarray): Input features - should be a tuple of (rgb_frames, thermal_frames)
            y (np.ndarray): Target values
            **kwargs: Additional arguments for evaluation

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Evaluate the model
        metrics = self.model.evaluate(X, y, verbose=0)

        # Create a dictionary of metrics
        metrics_dict = {
            "loss": metrics[0],
            "mse": metrics[1],
            "mae": metrics[2]
        }

        return metrics_dict

    def save(self, path: str) -> None:
        """
        Save the model to the given path.

        Args:
            path (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model
        self.model.save(path)

        logging.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'TensorFlowDualStreamCNNLSTMModel':
        """
        Load a model from the given path.

        Args:
            path (str): Path to load the model from

        Returns:
            TensorFlowDualStreamCNNLSTMModel: Loaded model
        """
        # Load the model
        model = tf.keras.models.load_model(path)

        # Create a dummy instance (this is a bit of a hack, but necessary for the interface)
        # In a real implementation, you would also save and load the configuration
        dummy_input_shape = (10, 10)  # Placeholder
        dummy_config = {}

        instance = cls(dummy_input_shape, dummy_config)
        instance.model = model

        logging.info(f"Model loaded from {path}")
        return instance


class TensorFlowDualStreamCNNLSTMFactory(ModelFactory):
    """Factory for creating TensorFlow Dual-Stream CNN-LSTM models."""

    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a TensorFlow Dual-Stream CNN-LSTM model."""
        return TensorFlowDualStreamCNNLSTMModel(input_shape, config)


# Register the model factories
ModelRegistry.register("tensorflow_cnn", TensorFlowCNNFactory())
ModelRegistry.register("tensorflow_cnn_lstm", TensorFlowCNNLSTMFactory())
ModelRegistry.register("tensorflow_dual_stream_cnn_lstm", TensorFlowDualStreamCNNLSTMFactory())
