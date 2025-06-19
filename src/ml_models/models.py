# src/ml_models/models.py

import logging

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

# --- Setup logging for this module ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(module)s - %(message)s",
)


def build_lstm_model(input_shape: tuple, learning_rate: float = 0.001) -> Sequential:
    """
    Builds and compiles a sequential LSTM model for GSR prediction.

    This model architecture is designed for time-series regression. It consists of
    two LSTM layers to capture temporal dependencies in the input signal windows,
    followed by dense layers for regression to a single output value. Dropout is
    included to prevent overfitting.

    Args:
        input_shape (tuple): The shape of the input data, which should be
                             (window_size, num_features).
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        Sequential: A compiled, untrained Keras model.
    """
    logging.info(f"Building LSTM model with input shape: {input_shape}")

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(units=64, return_sequences=True),
            Dropout(0.2),
            LSTM(units=32, return_sequences=False),
            Dropout(0.2),
            Dense(units=16, activation="relu"),
            Dense(units=1, activation="linear"),
        ]
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, loss="mean_absolute_error", metrics=["mean_squared_error"]
    )

    logging.info("LSTM model built and compiled successfully.")
    model.summary()
    return model


def build_ae_model(
    input_shape: tuple, latent_dim: int = 32, learning_rate: float = 0.001
) -> Model:
    """
    Builds and compiles a simple Autoencoder (AE) model for time-series data.

    This model is useful for learning compressed representations (encodings) of the
    input data in an unsupervised manner. It consists of an LSTM encoder and an LSTM decoder.

    Args:
        input_shape (tuple): The shape of the input data (window_size, num_features).
        latent_dim (int): The dimensionality of the compressed latent space.
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        Model: A compiled, untrained Keras Autoencoder model.
    """
    logging.info(
        f"Building Autoencoder model with input shape: {input_shape} and latent dim: {latent_dim}"
    )

    # --- Encoder ---
    encoder_inputs = Input(shape=input_shape)
    # Flatten the time series to process with Dense layers
    x = Flatten()(encoder_inputs)
    x = Dense(128, activation="relu")(x)
    # Latent space representation
    latent_space = Dense(latent_dim, activation="relu")(x)
    encoder = Model(encoder_inputs, latent_space, name="encoder")

    # --- Decoder ---
    decoder_inputs = Input(shape=(latent_dim,))
    x = Dense(128, activation="relu")(decoder_inputs)
    # Reconstruct the flattened shape
    reconstructed_flat = Dense(input_shape[0] * input_shape[1], activation="sigmoid")(x)
    # Reshape back to original input shape
    reconstructed = Reshape(input_shape)(reconstructed_flat)
    decoder = Model(decoder_inputs, reconstructed, name="decoder")

    # --- Autoencoder (Encoder + Decoder) ---
    autoencoder = Model(
        encoder_inputs, decoder(encoder(encoder_inputs)), name="autoencoder"
    )

    optimizer = Adam(learning_rate=learning_rate)
    autoencoder.compile(optimizer=optimizer, loss="mse")

    logging.info("Autoencoder model built and compiled successfully.")
    autoencoder.summary()
    return autoencoder


def build_vae_model(
    input_shape: tuple, latent_dim: int = 32, learning_rate: float = 0.001
) -> Model:
    """
    Builds and compiles a Variational Autoencoder (VAE) for time-series data.

    A VAE learns a probabilistic latent space, making it useful for generative tasks
    and understanding the data distribution.

    Args:
        input_shape (tuple): The shape of the input data (window_size, num_features).
        latent_dim (int): The dimensionality of the latent space.
        learning_rate (float): The learning rate for the Adam optimizer.

    Returns:
        Model: A compiled, untrained Keras VAE model.
    """
    logging.info(
        f"Building VAE model with input shape: {input_shape} and latent dim: {latent_dim}"
    )
    original_dim = input_shape[0] * input_shape[1]

    # --- Encoder ---
    encoder_inputs = Input(shape=input_shape)
    x = Flatten()(encoder_inputs)
    x = Dense(128, activation="relu")(x)
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
    x = Dense(128, activation="relu")(decoder_inputs)
    reconstructed_flat = Dense(original_dim, activation="sigmoid")(x)
    reconstructed = Reshape(input_shape)(reconstructed_flat)
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

    optimizer = Adam(learning_rate=learning_rate)
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

    print("\n--- Testing LSTM Model Creation ---")
    lstm_model = build_lstm_model(input_shape=example_input_shape)

    print("\n--- Testing Autoencoder Model Creation ---")
    ae_model = build_ae_model(input_shape=example_input_shape, latent_dim=16)

    print("\n--- Testing Variational Autoencoder Model Creation ---")
    vae_model = build_vae_model(input_shape=example_input_shape, latent_dim=16)
