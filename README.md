# Fashion Analysis AI App

## Overview
This application analyzes fashion characteristics from an image using computer vision and machine learning.

The system integrates two independent pipelines:

1. Color Analysis  
   - Average color extraction  
   - Personal color classification (4 seasons) using RandomForest

2. Shape Analysis  
   - Contour extraction using OpenCV  
   - Clothing silhouette classification using YOLOv8

The application is implemented with Streamlit and provides visual explanations using Altair.

## Tech Stack
- Python
- Streamlit
- OpenCV
- RandomForest (scikit-learn)
- YOLOv8
- Altair

## Run Locally

pip install -r requirements.txt  
streamlit run fassion_app6.py
