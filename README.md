# Main-repository

# Classification model
## Prerequirements

RAF-DB dataset installed and in the same folder 

## Environment setup using conda:

Step 1: Create Conda Environment 
conda create -n fer_cls python=3.10 -y	

Step 2: Install packages for model itself: 
conda activate fer_cls
conda install pytorch torchvision torchaudio -c pytorch
conda install numpy pillow

Step 3 (extra): Install packages for confusion matrix and plots:
conda install matplotlib scikit-learn

## Output

The classification model outputs a trained model best_model_cosine.pt


# Video Demo: Facial Emotion Recognition with Grad-CAM

This program processes a video file to detect faces, predict emotions using our CNN model, and overlay **Grad-CAM heatmaps** along with predicted emotion labels on each detected face. The processed video is saved as output.

---

## Features

- Grad-CAM visualization for model interpretability
- Saves output video with overlaid heatmaps and emotion labels
- Supports common video formats like `.mp4` and `.avi`

---

## Supported Emotions

- Surprise  
- Fear  
- Disgust  
- Happiness  
- Sadness  
- Anger  

---

## Environment Setup

Install the required Python packages:


pip install torch torchvision opencv-python numpy pytorch-grad-cam

## Usage

1. Place input video in the project directory or provide a full path.

2. Run the Program:
python video_demo/demo.py

3. Enter the input and output paths
4. The processed video with overlaid Grad-CAM and predicted emotion labels will be saved at the output path.
