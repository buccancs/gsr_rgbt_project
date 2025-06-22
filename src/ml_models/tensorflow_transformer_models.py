# src/ml_models/tensorflow_transformer_models.py

"""
TensorFlow Transformer model implementations for time series regression.

This module contains TensorFlow implementations of Transformer models for time series data,
including positional encoding and multi-head attention mechanisms.
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


class PositionalEncoding(layers.Layer):
    """
    TensorFlow implementation of positional encoding for Transformer models.
    
    This layer adds positional information to the input embeddings to help
    the model understand the order of the sequence.
    """

    def __init__(self, d_model: int, max_len: int = 5000, **kwargs):
        """
        Initialize the positional encoding layer.

        Args:
            d_model (int): The dimension of the model (embedding size)
            max_len (int): Maximum length of the input sequences
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        
        # Create div_term for the sinusoidal pattern
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = np.sin(position * div_term)
        
        # Apply cos to odd indices
        if d_model % 2 == 1:
            pe[:, 1::2] = np.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = np.cos(position * div_term)
        
        # Add batch dimension and convert to tensor
        self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
    
    def call(self, inputs):
        """
        Add positional encoding to the input.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Input with positional encoding added
        """
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pe[:, :seq_len, :]


class TensorFlowTransformer(Model):
    """
    TensorFlow Transformer model for time series regression.

    This model uses multi-head self-attention mechanisms to capture
    long-range dependencies in time series data.
    """

    def __init__(self, input_size: int, d_model: int, nhead: int, 
                 num_layers: int, dim_feedforward: int, dropout: float,
                 fc_layers: List[int], activations: List[str]):
        """
        Initialize the Transformer model.

        Args:
            input_size (int): Size of input features
            d_model (int): Dimension of the model (embedding size)
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dim_feedforward (int): Dimension of the feedforward network
            dropout (float): Dropout rate
            fc_layers (List[int]): Sizes of fully connected layers
            activations (List[str]): Activation functions for each layer
        """
        super(TensorFlowTransformer, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.fc_layers = fc_layers
        self.activations = activations
        
        # Input projection layer
        self.input_projection = layers.Dense(d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Dropout layer
        self.dropout_layer = layers.Dropout(dropout)
        
        # Transformer encoder layers
        self.transformer_layers = []
        for _ in range(num_layers):
            # Multi-head attention
            attention_layer = layers.MultiHeadAttention(
                num_heads=nhead,
                key_dim=d_model // nhead,
                dropout=dropout
            )
            
            # Feed-forward network
            ffn = tf.keras.Sequential([
                layers.Dense(dim_feedforward, activation='relu'),
                layers.Dropout(dropout),
                layers.Dense(d_model)
            ])
            
            # Layer normalization
            norm1 = layers.LayerNormalization(epsilon=1e-6)
            norm2 = layers.LayerNormalization(epsilon=1e-6)
            
            # Dropout layers
            dropout1 = layers.Dropout(dropout)
            dropout2 = layers.Dropout(dropout)
            
            self.transformer_layers.append({
                'attention': attention_layer,
                'ffn': ffn,
                'norm1': norm1,
                'norm2': norm2,
                'dropout1': dropout1,
                'dropout2': dropout2
            })
        
        # Global average pooling
        self.global_avg_pool = layers.GlobalAveragePooling1D()
        
        # Fully connected layers
        self.fc_layers_list = []
        for i, units in enumerate(fc_layers):
            self.fc_layers_list.append(layers.Dense(units))
            
            # Add activation
            if i < len(activations):
                if activations[i] == 'relu':
                    self.fc_layers_list.append(layers.ReLU())
                elif activations[i] == 'tanh':
                    self.fc_layers_list.append(layers.Activation('tanh'))
                elif activations[i] == 'sigmoid':
                    self.fc_layers_list.append(layers.Activation('sigmoid'))
                elif activations[i] == 'leaky_relu':
                    self.fc_layers_list.append(layers.LeakyReLU(0.1))
                elif activations[i] == 'linear':
                    # Linear activation is the default in TensorFlow
                    pass
            
            # Add dropout after activation (except for the last layer)
            if i < len(fc_layers) - 1 and dropout > 0:
                self.fc_layers_list.append(layers.Dropout(dropout))
    
    def call(self, inputs, training=None):
        """
        Forward pass of the Transformer model.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, features)
            training: Boolean indicating whether the model is in training mode

        Returns:
            Tensor: Output tensor
        """
        # Project input to d_model dimensions
        x = self.input_projection(inputs)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout_layer(x, training=training)
        
        # Pass through transformer layers
        for layer_dict in self.transformer_layers:
            # Multi-head self-attention
            attn_output = layer_dict['attention'](x, x, training=training)
            attn_output = layer_dict['dropout1'](attn_output, training=training)
            
            # Add & Norm
            x = layer_dict['norm1'](x + attn_output)
            
            # Feed-forward network
            ffn_output = layer_dict['ffn'](x, training=training)
            ffn_output = layer_dict['dropout2'](ffn_output, training=training)
            
            # Add & Norm
            x = layer_dict['norm2'](x + ffn_output)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Pass through fully connected layers
        for layer in self.fc_layers_list:
            x = layer(x)
        
        return x


class TensorFlowTransformerModel(BaseModel):
    """
    TensorFlow Transformer model implementation of the BaseModel interface.
    """

    def __init__(self, input_shape: Tuple[int, int], config: Dict[str, Any]):
        """
        Initialize the TensorFlow Transformer model.

        Args:
            input_shape (Tuple[int, int]): Shape of the input data (window_size, features)
            config (Dict[str, Any]): Model configuration parameters
        """
        self.input_shape = input_shape
        self.config = config

        # Extract model parameters
        model_params = config.get("model_params", {})
        input_size = input_shape[1]  # Number of features

        d_model = model_params.get("d_model", 128)
        nhead = model_params.get("nhead", 8)
        num_layers = model_params.get("num_layers", 6)
        dim_feedforward = model_params.get("dim_feedforward", 512)
        dropout = model_params.get("dropout", 0.1)
        fc_layers = model_params.get("fc_layers", [64, 32, 1])
        activations = model_params.get("activations", ["relu", "relu", "linear"])

        # Create the model
        self.model = TensorFlowTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            fc_layers=fc_layers,
            activations=activations
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
    def load(cls, path: str) -> 'TensorFlowTransformerModel':
        """
        Load a model from the given path.

        Args:
            path (str): Path to load the model from

        Returns:
            TensorFlowTransformerModel: Loaded model
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


class TensorFlowTransformerFactory(ModelFactory):
    """Factory for creating TensorFlow Transformer models."""
    
    def create_model(self, input_shape: Tuple, config: Dict[str, Any]) -> BaseModel:
        """Create a TensorFlow Transformer model."""
        return TensorFlowTransformerModel(input_shape, config)


# Register the model factory
ModelRegistry.register("tensorflow_transformer", TensorFlowTransformerFactory())