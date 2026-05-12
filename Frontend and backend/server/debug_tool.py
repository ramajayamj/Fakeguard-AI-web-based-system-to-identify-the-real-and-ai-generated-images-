import os
import tensorflow as tf
import numpy as np
import io
from PIL import Image
import sys

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'fake_refund_image_detector_efficientnetv2.keras')

def preprocess_image(path):
    try:
        image = Image.open(path).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = image_array.astype('float32') / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_tool.py <path_to_image_or_directory>")
        return

    target_path = sys.argv[1]
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")

    files = []
    if os.path.isfile(target_path):
        files.append(target_path)
    elif os.path.isdir(target_path):
        for root, _, filenames in os.walk(target_path):
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    files.append(os.path.join(root, f))
    
    print(f"Found {len(files)} images.")
    print("-" * 50)
    print(f"{'Filename':<30} | {'Raw Score':<10} | {'Prediction':<10}")
    print("-" * 50)

    for f in files:
        img = preprocess_image(f)
        if img is not None:
            pred = model.predict(img, verbose=0)[0][0]
            # 0 = Fake, 1 = Real
            label = "Real" if pred >= 0.5 else "Fake"
            print(f"{os.path.basename(f):<30} | {pred:.4f}     | {label}")

if __name__ == "__main__":
    main()
