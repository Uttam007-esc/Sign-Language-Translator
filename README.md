#Sign Language Translator using Deep Learning

## Overview

AI Sign Language Translator is a Deep Learning-based application designed to bridge the communication gap between hearing-impaired individuals and the wider community. The system recognizes sign language hand gestures and translates them into readable text in real time using Computer Vision and Deep Learning techniques.

This project utilizes Convolutional Neural Networks (CNNs) for gesture classification and integrates OpenCV for image processing and webcam-based gesture detection.

## Problem Statement

Communication between hearing-impaired individuals and people unfamiliar with sign language remains a challenge. This project aims to provide an intelligent solution that recognizes sign language gestures and converts them into text, enabling seamless communication.

## Features

* Real-time hand gesture recognition
* Sign language alphabet detection
* Deep Learning-based gesture classification
* Webcam integration using OpenCV
* Image preprocessing and augmentation
* Scalable architecture for future word and sentence recognition
* Potential integration with Text-to-Speech systems

## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* MediaPipe
* cvzone
* Scikit-Learn

## System Architecture

Webcam Input → Image Preprocessing → Hand Detection → CNN Model Prediction → Text Output

## Dataset

The model was trained using sign language datasets including:

* ASL Alphabet Dataset
* Sign Language MNIST

The dataset images were resized, normalized, and augmented before training.

## Deep Learning Model

Model Type: Convolutional Neural Network (CNN)

Key Layers:

* Convolution Layers
* Batch Normalization
* Max Pooling Layers
* Dense Layers
* Dropout Layer
* Softmax Output Layer

## Project Structure

```text
data/
models/
src/
screenshots/
requirements.txt
README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python app.py
```

## Results

The system successfully recognizes sign language gestures and translates them into corresponding alphabets in real time.

## Future Enhancements

* Word-level recognition
* Sentence generation
* Voice output integration
* Mobile application deployment
* CNN + LSTM architecture for dynamic gestures

## Team Members

* Uttam Kumar Dingari
* D. Shanmukh Nayan
* K. Sai Charan Reddy

## Guide

Mr. G. Srihari Babu
Senior Assistant Professor
GCET
