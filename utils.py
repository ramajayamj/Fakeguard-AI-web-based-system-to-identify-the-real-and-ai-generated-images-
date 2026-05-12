import cv2
import numpy as np
import io
from PIL import Image

def preprocess_image(image_bytes, target_size=64):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image = image.resize((target_size, target_size))
        image_array = np.array(image).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        print(f"Image preprocess error: {e}")
        return None

def process_video_frames(video_path, target_size=64, max_frames=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // max_frames)
    count = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame).resize((target_size, target_size))
            image_array = np.array(image).astype('float32') / 255.0
            frames.append(image_array)
            frame_count += 1
            if frame_count >= max_frames:
                break
        count += 1

    cap.release()
    return np.array(frames) if frames else None
