# models.py
import tensorflow as tf
from tensorflow.keras import layers, models
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_baseline_model(input_dim):
    """Build a simple MLP for regression."""
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)  # Regression output
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    logging.info("Baseline model built.")
    return model

class AttentionLayer(layers.Layer):
    """Custom attention layer for spectral data."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch_size, features)
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Compute attention scores
        e = tf.tanh(tf.matmul(inputs, self.W) + self.b)  # (batch_size, features, 1)
        alpha = tf.nn.softmax(e, axis=1)  # (batch_size, features, 1)
        context = inputs * alpha  # (batch_size, features) * (batch_size, features, 1) -> (batch_size, features)
        # No reduce_sum here; keep the feature dimension
        return context

def build_attention_model(input_dim):
    """Build a neural network with an attention mechanism."""
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inputs)  # (batch_size, input_dim, 1)
    x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(x)  # (batch_size, input_dim, 32)
    x = layers.MaxPooling1D(pool_size=2)(x)  # (batch_size, input_dim//2, 32)
    x = layers.Flatten()(x)  # (batch_size, (input_dim//2) * 32)
    x = AttentionLayer()(x)  # (batch_size, (input_dim//2) * 32)
    x = layers.Dense(64, activation='relu')(x)  # (batch_size, 64)
    x = layers.Dropout(0.3)(x)  # (batch_size, 64)
    outputs = layers.Dense(1)(x)  # (batch_size, 1)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    logging.info("Attention model built.")
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=16):
    """Train the model with validation."""
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), verbose=1)
    logging.info("Model training completed.")
    return history

if __name__ == "__main__":
    # Quick test
    import numpy as np
    input_dim = 448  # Adjust based on your data
    model = build_attention_model(input_dim)
    model.summary()
    dummy_data = np.random.rand(10, input_dim)
    output = model.predict(dummy_data)
    print("Output shape:", output.shape)  # Should be (10, 1)