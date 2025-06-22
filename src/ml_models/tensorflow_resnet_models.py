# src/ml_models/tensorflow_resnet_models.py

"""
TensorFlow ResNet model implementations for time series regression.

This module contains TensorFlow implementations of ResNet models for time series data,
including residual blocks and ResNet architectures.
"""

import logging
import os
from typing import Dict, Any, Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Model

from src.ml_models.model_interface import BaseModel, ModelFactory, ModelRegistry

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


class ResidualBlock(layers.Layer):
    """
    TensorFlow implementation of a residual block for ResNet.
    
    This block implements the skip connection that allows gradients to flow
    directly through the network, enabling training of very deep networks.
    """

    def __init__(self, filters: int, kernel_size: int = 3, stride: int = 1, 
                 activation: str = 'relu', **kwargs):
        """
        Initialize the residual block.

        Args:
            filters (int): Number of filters in the convolutional layers
            kernel_size (int): Size of the convolutional kernel
            stride (int): Stride for the first convolution
            activation (str): Activation function to use
        """
        super(ResidualBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        
        # First convolution
        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            use_bias=False
        )
        self.bn1 = layers.BatchNormalization()
        
        # Second convolution
        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            use_bias=False
        )
        self.bn2 = layers.BatchNormalization()
        
        # Activation layer
        if activation == 'relu':
            self.activation_layer = layers.ReLU()
        elif activation == 'tanh':
            self.activation_layer = layers.Activation('tanh')
        elif activation == 'sigmoid':
            self.activation_layer = layers.Activation('sigmoid')
        elif activation == 'leaky_relu':
            self.activation_layer = layers.LeakyReLU(0.1)
        else:
            self.activation_layer = layers.ReLU()  # Default
        
        # Shortcut connection (if needed)
        self.shortcut = None
        
    def build(self, input_shape):
        """Build the layer based on input shape."""
        super(ResidualBlock, self).build(input_shape)
        
        # Check if we need a shortcut connection
        input_filters = input_shape[-1]
        if input_filters != self.filters or self.stride != 1:
            self.shortcut = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                strides=self.stride,
                padding='same',
                use_bias=False
            )
            self.shortcut_bn = layers.BatchNormalization()
    
    def call(self, inputs, training=None):
        """
        Forward pass of the residual block.

        Args:
            inputs: Input tensor
            training: Boolean indicating whether the model is in training mode

        Returns:
            Tensor: Output tensor
        """
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation_layer(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Shortcut connection
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
        else:
            shortcut = inputs
        
        # Add shortcut and apply activation
        x = layers.Add()([x, shortcut])
        x = self.activation_layer(x)
        
        return x


class TensorFlowResNet(Model):
    """
    TensorFlow ResNet model for time series regression.

    This model uses residual connections to enable training of deep networks
    for time series data processing.
    """

    def __init__(self, input_channels: int, layers_config: List[int], 
                 blocks_per_layer: List[int], fc_layers: List[int],
                 activations: List[str], dropout_rate: float = 0.2):
        """
        Initialize the ResNet model.

        Args:
            input_channels (int): Number of input channels (features)
            layers_config (List[int]): Number of filters for each layer
            blocks_per_layer (List[int]): Number of residual blocks per layer
            fc_layers (List[int]): Sizes of fully connected layers
            activations (List[str]): Activation functions for each layer
            dropout_rate (float, optional): Dropout rate for FC layers
        """
        super(TensorFlowResNet, self).__init__()
        
        # Validate input parameters
        assert len(layers_config) == len(blocks_per_layer), \
            "layers_config and blocks_per_layer must have the same length"
        
        self.layers_config = layers_config
        self.blocks_per_layer = blocks_per_layer
        self.fc_layers = fc_layers
        self.activations = activations
        self.dropout_rate = dropout_rate
        
        # Initial convolution
        self.initial_conv = layers.Conv1D(
            filters=layers_config[0],
            kernel_size=7,
            strides=2,
            padding='same',
            use_bias=False
        )
        self.initial_bn = layers.BatchNormalization()
        self.initial_activation = layers.ReLU()
        self.initial_pool = layers.MaxPool1D(pool_size=3, strides=2, padding='same')
        
        # Residual layers
        self.residual_layers = []
        current_filters = layers_config[0]
        
        for i, (filters, blocks) in enumerate(zip(layers_config, blocks_per_layer)):
            # First block of each layer may have stride 2 (except the first layer)
            stride = 2 if i > 0 else 1
            
            # Create blocks for this layer
            layer_blocks = []
            for j in range(blocks):
                block_stride = stride if j == 0 else 1
                activation = activations[i] if i < len(activations) else 'relu'
                
                block = ResidualBlock(
                    filters=filters,
                    stride=block_stride,
                    activation=activation
                )
                layer_blocks.append(block)
            
            self.residual_layers.append(layer_blocks)
            current_filters = filters
        
        # Global average pooling
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        
        # Fully connected layers
        self.fc_layers_list = []
        for i, units in enumerate(fc_layers):
            self.fc_layers_list.append(layers.Dense(units))
            
            # Add activation
            activation_idx = i + len(layers_config)
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
        Forward pass of the ResNet model.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features)
            training: Boolean indicating whether the model is in training mode

        Returns:
            Tensor: Output tensor
        """
        # Initial convolution
        x = self.initial_conv(inputs)
        x = self.initial_bn(x, training=training)
        x = self.initial_activation(x)
        x = self.initial_pool(x)
        
        # Pass through residual layers
        for layer_blocks in self.residual_layers:
            for block in layer_blocks:
                x = block(x, training=training)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Pass through fully connected layers
        for layer in self.fc_layers_list:
            x = layer(x)
        
        return x


class TensorFlowResNetModel(BaseModel):
    """
    TensorFlow ResNet model implementation of the BaseModel interface.
    """

    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the TensorFlow ResNet model.

        Args:
            input_shape (Tuple[int, int]): Shape of the input data (window_size, features)
            config (Dict[str, Any]): Model configuration parameters
        """
        self.input_shape = input_shape
        self.config = config

        # Extract model parameters
        model_params = config.get("model_params", {})
        input_channels = input_shape[1]  # Number of features

        layers_config = model_params.get("layers", [64, 128, 256, 512])
        blocks_per_layer = model_params.get("blocks_per_layer", [2, 2, 2, 2])
        fc_layers = model_params.get("fc_layers", [64, 32, 1])
        activations = model_params.get("activations", ["relu"] * (len(layers_config) + len(fc_layers)))
        dropout_rate = model_params.get("dropout_rate", 0.2)

        # Create the model
        self.model = TensorFlowResNet(
            input_channels=input_channels,
            layers_config=layers_config,
            blocks_per_layer=blocks_per_layer,
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
    def load(cls, path: str) -> 'TensorFlowResNetModel':
        """
        Load a model from the given path.

        Args:
            path (str): Path to load the model from

        Returns:
            TensorFlowResNetModel: Loaded model
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


class TensorFlowResNetFactory(ModelFactory):
    """Factory for creating TensorFlow ResNet models."""
    
    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a TensorFlow ResNet model."""
        return TensorFlowResNetModel(input_shape, config)


# Register the model factory
ModelRegistry.register("tensorflow_resnet", TensorFlowResNetFactory())