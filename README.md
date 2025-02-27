# Crop Disease Detection

This repository contains the code for detecting crop types and diseases using Convolutional Neural Networks (CNNs) and Hugging Face Transformers. The solution integrates these models into a user-friendly Streamlit application, allowing users to upload images and receive real-time diagnostic results.

## Overview

This project aims to enhance precision agriculture by providing early and accurate detection of crop diseases. By harnessing the power of AI, farmers and agricultural researchers can make informed decisions, reduce crop damage, and improve yields.

## Features

- **Data Loading and Preprocessing**: Functions to load and preprocess images.
- **CNN Model**: A Convolutional Neural Network for crop classification.
- **Vision Transformer (ViT) Model**: A Hugging Face Transformer for disease detection.
- **Streamlit Application**: A web app to upload images and get real-time classification and detection results.

## Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/crop-disease-detection.git
   cd crop-disease-detection
   
## Install Required Libraries:
pip install tensorflow transformers datasets torch torchvision streamlit

## Run the Streamlit App:
streamlit run app.py
