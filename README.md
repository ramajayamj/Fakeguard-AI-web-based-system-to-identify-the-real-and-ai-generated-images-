# FakeGuard AI

## Overview

FakeGuard AI is a web-based deep learning application developed to detect fake or AI-generated images used in online fraud scenarios such as fraudulent refund claims in quick-commerce and e-commerce platforms.

The system combines a hybrid CNN–Vision Transformer (CNN–ViT) model with a lightweight Flask framework to provide real-time image authenticity verification through an easy-to-use web interface.

The project aims to reduce fraudulent activities by automatically classifying uploaded images as **Real** or **Fake**, helping digital platforms improve trust, security, and operational efficiency.

---

# Features

- Real-time fake image detection
- Hybrid CNN + Vision Transformer architecture
- Flask-based web application
- User-friendly image upload interface
- Real/Fake prediction visualization
- Chatbot-assisted interaction
- Scalable and modular design

---

# Tech Stack

## Frontend
- HTML
- CSS
- JavaScript

## Backend
- Flask (Python)

## Deep Learning
- PyTorch

## Model Architecture
- CNN + Vision Transformer (ViT)

## Libraries Used
- NumPy
- OpenCV
- TorchVision
- PIL

---

# Model Architecture

The proposed hybrid model combines:

## Convolutional Neural Network (CNN)

Extracts local image features such as:
- Textures
- Edges
- Manipulation artifacts

## Vision Transformer (ViT)

Captures global contextual relationships using self-attention mechanisms to identify structural inconsistencies in images.

This hybrid approach improves detection accuracy by combining both local and global feature learning.

---

# Workflow

1. User uploads an image through the web interface  
2. Flask backend receives the image  
3. Image preprocessing is performed:
   - Resizing
   - Normalization
4. Preprocessed image is passed to the hybrid CNN–ViT model  
5. Model predicts whether the image is:
   - Real
   - Fake
6. Result is displayed instantly on the frontend  

---

# Dataset

The model is trained using publicly available datasets containing:

- Real images
- AI-generated / manipulated images

---

# Performance Metrics

The system is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix Analysis

The hybrid model achieved high classification performance with stable convergence during training and testing.

---

# Future Enhancements

- Mobile application integration
- Video deepfake detection
- Cloud deployment
- Multi-language chatbot support
- Advanced fraud analytics dashboard

---

# Author

Ramajayam
