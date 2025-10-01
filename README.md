# Speech Emotion Recognition

A deep learning-based system that automatically identifies emotions from speech audio using Convolutional Neural Networks. The model analyzes audio features to classify speech into 8 distinct emotional categories: neutral, calm, happy, sad, angry, fear, disgust, and surprise.

## Overview

Speech emotion recognition is crucial for applications in human-computer interaction, customer service analysis, mental health monitoring, and voice-based assistants. This project leverages deep learning to extract meaningful patterns from audio signals and accurately classify the underlying emotional state of the speaker.

The system processes raw audio files, extracts relevant acoustic features, and uses a trained CNN model to predict emotions with high accuracy across multiple datasets.

## Features

- **Multi-Dataset Integration**: Combines four major speech emotion datasets (RAVDESS, CREMA-D, TESS, SAVEE) totaling ~14,000+ samples after augmentation
- **Comprehensive Feature Extraction**: 
  - Zero Crossing Rate (ZCR) - Signal frequency content
  - Chroma STFT - Pitch class representation
  - MFCCs - Spectral envelope characteristics
  - RMS Energy - Audio loudness measurement
  - Mel-Spectrogram - Time-frequency representation
- **Data Augmentation**: Implements four augmentation techniques to improve model robustness:
  - Noise injection for handling noisy environments
  - Time stretching for speech rate variations
  - Pitch shifting for different voice ranges
  - Time shifting for temporal variations
- **Deep CNN Architecture**: Multi-layer Conv1D network optimized for sequential audio data with dropout regularization to prevent overfitting

## Technologies

**Core:** Python 3.x | TensorFlow | Keras  
**Audio Processing:** Librosa  
**Data Science:** NumPy | Pandas | Scikit-learn  
**Visualization:** Matplotlib | Seaborn

## Datasets

1. **RAVDESS** - Ryerson Audio-Visual Database of Emotional Speech and Song
2. **CREMA-D** - Crowd-sourced Emotional Multimodal Actors Dataset
3. **TESS** - Toronto Emotional Speech Set
4. **SAVEE** - Surrey Audio-Visual Expressed Emotion

Each dataset provides diverse speakers, accents, and recording conditions, ensuring the model generalizes well across different scenarios.

The architecture uses 1D convolutions to capture temporal patterns in audio features, with progressive dimensionality reduction through pooling layers.

## Installation
```bash
# Clone the repository
git clone https://github.com/jais-Arush43/Speech_emotion_recognition.git
cd Speech_emotion_recognition

# Install dependencies
pip install tensorflow keras librosa scikit-learn numpy pandas matplotlib seaborn
