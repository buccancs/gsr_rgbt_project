# src/ml_models/models.py

import logging
from typing import Dict, Any, Union, List, Optional

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    Flatten,
    Reshape,
    Lambda,
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

from src.ml_models.model_config import ModelConfig

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def build_lstm_model(input_shape: tuple, config: Optional[ModelConfig] = None) -> Sequential:
    """
    Builds and compiles a sequential LSTM model for GSR prediction.

    This model architecture is designed for time-series regression. It consists of
    LSTM layers to capture temporal dependencies in the input signal windows,
    followed by dense layers for regression to a single output value. Dropout is
    included to prevent overfitting.

    Args:
        input_shape (tuple): The shape of the input data, which should be
                             (window_size, num_features).
        config (ModelConfig, optional): Configuration object containing model parameters.
                                       If None, default LSTM config will be used.

    Returns:
        Sequential: A compiled, untrained Keras model.
    """
    if config is None:
        config = ModelConfig("lstm")

    model_config = config.get_config()
    logging.info(f"Building LSTM model with input shape: {input_shape}")

    # Create the model with an input layer
    model = Sequential([Input(shape=input_shape)])

    # Add layers based on configuration
    for layer_config in model_config.get("layers", []):
        layer_type = layer_config.get("type", "").lower()

        if layer_type == "lstm":
            model.add(LSTM(
                units=layer_config.get("units", 64),
                return_sequences=layer_config.get("return_sequences", False),
                activation=layer_config.get("activation", "tanh")
            ))
        elif layer_type == "dropout":
            model.add(Dropout(rate=layer_config.get("rate", 0.2)))
        elif layer_type == "dense":
            model.add(Dense(
                units=layer_config.get("units", 16),
                activation=layer_config.get("activation", "relu")
            ))

    # Compile the model
    compile_params = config.get_compile_params()
    optimizer_config = compile_params.get("optimizer", {})
    optimizer_type = optimizer_config.get("type", "adam").lower()

    if optimizer_type == "adam":
        optimizer = Adam(learning_rate=optimizer_config.get("learning_rate", 0.001))
    else:
        # Default to Adam if the optimizer type is not recognized
        optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=compile_params.get("loss", "mean_absolute_error"),
        metrics=compile_params.get("metrics", ["mean_squared_error"])
    )

    logging.info("LSTM model built and compiled successfully.")
    model.summary()
    return model


def build_ae_model(
    input_shape: tuple, config: Optional[ModelConfig] = None
) -> Model:
    """
    Builds and compiles a simple Autoencoder (AE) model for time-series data.

    This model is useful for learning compressed representations (encodings) of the
    input data in an unsupervised manner. It consists of an encoder and a decoder.

    Args:
        input_shape (tuple): The shape of the input data (window_size, num_features).
        config (ModelConfig, optional): Configuration object containing model parameters.
                                       If None, default autoencoder config will be used.

    Returns:
        Model: A compiled, untrained Keras Autoencoder model.
    """
    if config is None:
        config = ModelConfig("autoencoder")

    model_config = config.get_config()
    latent_dim = model_config.get("latent_dim", 32)

    logging.info(
        f"Building Autoencoder model with input shape: {input_shape} and latent dim: {latent_dim}"
    )

    # --- Encoder ---
    encoder_inputs = Input(shape=input_shape)
    x = encoder_inputs

    # Process encoder layers from config
    for layer_config in model_config.get("encoder_layers", []):
        layer_type = layer_config.get("type", "").lower()

        if layer_type == "flatten":
            x = Flatten()(x)
        elif layer_type == "dense":
            units = layer_config.get("units", 128)
            # Handle special case for latent_dim
            if units == "latent_dim":
                units = latent_dim
            x = Dense(units=units, activation=layer_config.get("activation", "relu"))(x)

    # The output of the encoder is the latent space representation
    latent_space = x
    encoder = Model(encoder_inputs, latent_space, name="encoder")

    # --- Decoder ---
    decoder_inputs = Input(shape=(latent_dim,))
    x = decoder_inputs

    # Process decoder layers from config
    for layer_config in model_config.get("decoder_layers", []):
        layer_type = layer_config.get("type", "").lower()

        if layer_type == "dense":
            units = layer_config.get("units", 128)
            # Handle special case for input_size
            if units == "input_size":
                units = input_shape[0] * input_shape[1]
            x = Dense(units=units, activation=layer_config.get("activation", "relu"))(x)
        elif layer_type == "reshape":
            target_shape = layer_config.get("target_shape", "input_shape")
            # Handle special case for input_shape
            if target_shape == "input_shape":
                target_shape = input_shape
            x = Reshape(target_shape)(x)

    reconstructed = x
    decoder = Model(decoder_inputs, reconstructed, name="decoder")

    # --- Autoencoder (Encoder + Decoder) ---
    autoencoder = Model(
        encoder_inputs, decoder(encoder(encoder_inputs)), name="autoencoder"
    )

    # Compile the model
    compile_params = config.get_compile_params()
    optimizer_config = compile_params.get("optimizer", {})
    optimizer_type = optimizer_config.get("type", "adam").lower()

    if optimizer_type == "adam":
        optimizer = Adam(learning_rate=optimizer_config.get("learning_rate", 0.001))
    else:
        # Default to Adam if the optimizer type is not recognized
        optimizer = Adam(learning_rate=0.001)

    autoencoder.compile(
        optimizer=optimizer,
        loss=compile_params.get("loss", "mse")
    )

    logging.info("Autoencoder model built and compiled successfully.")
    autoencoder.summary()
    return autoencoder


def build_vae_model(
    input_shape: tuple, config: Optional[ModelConfig] = None
) -> Model:
    """
    Builds and compiles a Variational Autoencoder (VAE) for time-series data.

    A VAE learns a probabilistic latent space, making it useful for generative tasks
    and understanding the data distribution.

    Args:
        input_shape (tuple): The shape of the input data (window_size, num_features).
        config (ModelConfig, optional): Configuration object containing model parameters.
                                       If None, default VAE config will be used.

    Returns:
        Model: A compiled, untrained Keras VAE model.
    """
    if config is None:
        config = ModelConfig("vae")

    model_config = config.get_config()
    latent_dim = model_config.get("latent_dim", 32)

    logging.info(
        f"Building VAE model with input shape: {input_shape} and latent dim: {latent_dim}"
    )
    original_dim = input_shape[0] * input_shape[1]

    # --- Encoder ---
    encoder_inputs = Input(shape=input_shape)
    x = encoder_inputs

    # Process encoder layers from config
    for layer_config in model_config.get("encoder_layers", []):
        layer_type = layer_config.get("type", "").lower()

        if layer_type == "flatten":
            x = Flatten()(x)
        elif layer_type == "dense":
            units = layer_config.get("units", 128)
            # Handle special case for latent_dim
            if units == "latent_dim":
                units = latent_dim
            x = Dense(units=units, activation=layer_config.get("activation", "relu"))(x)

    # VAE specific layers for mean and log variance
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)

    # Sampling function (reparameterization trick)
    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # --- Decoder ---
    decoder_inputs = Input(shape=(latent_dim,), name="decoder_input")
    x = decoder_inputs

    # Process decoder layers from config
    for layer_config in model_config.get("decoder_layers", []):
        layer_type = layer_config.get("type", "").lower()

        if layer_type == "dense":
            units = layer_config.get("units", 128)
            # Handle special case for input_size
            if units == "input_size":
                units = original_dim
            x = Dense(units=units, activation=layer_config.get("activation", "relu"))(x)
        elif layer_type == "reshape":
            target_shape = layer_config.get("target_shape", "input_shape")
            # Handle special case for input_shape
            if target_shape == "input_shape":
                target_shape = input_shape
            x = Reshape(target_shape)(x)

    reconstructed = x
    decoder = Model(decoder_inputs, reconstructed, name="decoder")

    # --- VAE (Encoder + Decoder) ---
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = Model(encoder_inputs, outputs, name="vae")

    # --- VAE Loss Function ---
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            K.flatten(encoder_inputs), K.flatten(outputs)
        )
    )
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    # Compile the model
    compile_params = config.get_compile_params()
    optimizer_config = compile_params.get("optimizer", {})
    optimizer_type = optimizer_config.get("type", "adam").lower()

    if optimizer_type == "adam":
        optimizer = Adam(learning_rate=optimizer_config.get("learning_rate", 0.001))
    else:
        # Default to Adam if the optimizer type is not recognized
        optimizer = Adam(learning_rate=0.001)

    vae.compile(optimizer=optimizer)

    logging.info("VAE model built and compiled successfully.")
    vae.summary()
    return vae


# --- Example Usage ---
if __name__ == "__main__":
    # These values would typically come from the feature engineering step.
    example_window_size = 5 * 32
    example_num_features = 4
    example_input_shape = (example_window_size, example_num_features)

    from src.ml_models.model_config import ModelConfig, create_example_config_files
    from pathlib import Path

    # Create example configuration files
    config_dir = Path("./configs/models")
    create_example_config_files(config_dir)

    # Test with default configurations
    print("\n--- Testing LSTM Model Creation with Default Config ---")
    lstm_config = ModelConfig("lstm")
    lstm_model = build_lstm_model(input_shape=example_input_shape, config=lstm_config)

    print("\n--- Testing Autoencoder Model Creation with Default Config ---")
    ae_config = ModelConfig("autoencoder")
    # Update latent dimension
    ae_config.update_config({"latent_dim": 16})
    ae_model = build_ae_model(input_shape=example_input_shape, config=ae_config)

    print("\n--- Testing Variational Autoencoder Model Creation with Default Config ---")
    vae_config = ModelConfig("vae")
    # Update latent dimension
    vae_config.update_config({"latent_dim": 16})
    vae_model = build_vae_model(input_shape=example_input_shape, config=vae_config)

    print("\n--- Testing Model Creation with Custom Config ---")
    # Example of creating a custom LSTM configuration
    custom_lstm_config = ModelConfig("lstm")
    custom_lstm_config.update_config({
        "layers": [
            {"type": "lstm", "units": 128, "return_sequences": True},
            {"type": "dropout", "rate": 0.3},
            {"type": "lstm", "units": 64, "return_sequences": False},
            {"type": "dropout", "rate": 0.3},
            {"type": "dense", "units": 32, "activation": "relu"},
            {"type": "dense", "units": 1, "activation": "linear"}
        ],
        "compile_params": {
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.0005
            }
        }
    })
    custom_lstm_model = build_lstm_model(input_shape=example_input_shape, config=custom_lstm_config)
