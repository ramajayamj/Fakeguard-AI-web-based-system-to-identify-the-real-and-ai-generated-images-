import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
import numpy as np


@register_keras_serializable(package="Custom")
class PatchExtractor(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


@register_keras_serializable(package="Custom")
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })
        return config


_model = None


def load_model(model_path):
    global _model
    if _model is None:
        print(f"Loading CNN+ViT hybrid model from {model_path}...")
        _model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "PatchExtractor": PatchExtractor,
                "PatchEncoder": PatchEncoder,
            }
        )
        print("CNN+ViT model ready.")
    return _model


def predict(input_data):
    global _model
    if _model is None:
        raise Exception("Model not loaded.")

    predictions = _model.predict(input_data, verbose=0)
    mean_prediction = float(np.mean(predictions))

    predicted_class = "Real" if mean_prediction >= 0.5 else "Fake"
    confidence = mean_prediction if mean_prediction >= 0.5 else 1.0 - mean_prediction

    return predicted_class, confidence
