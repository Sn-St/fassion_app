# Fashion Analysis AI App

AI-powered fashion analysis application combining computer vision and machine learning.

This application analyzes clothing images and provides insights about how well the clothing may match a person's fashion characteristics, such as personal color and body silhouette.

---

# Overview

This project is a fashion analysis system built with Python and Streamlit.

The application analyzes clothing images and evaluates them from two perspectives:

* **Color compatibility**
* **Silhouette characteristics**

Two independent analysis pipelines extract visual features from the image and generate interpretable outputs.

---

# Project Motivation

I believe that the most enjoyable approach to fashion is simply wearing what you love.
At the same time, many people who enjoy fashion also want an additional layer of confidence — such as choosing clothes that make their skin look brighter or their body silhouette look more balanced.

This application is designed for people who already have a general idea of their personal color or body type but still wonder whether a specific outfit suits them.

For example, when finding a piece of clothing in an online shop or trying on clothes at home, people may ask:

*"This outfit looks cute, but does it actually suit me?"*

Rather than diagnosing a user's type from scratch, this application acts as a **support tool**.
It objectively analyzes clothing images and helps users check the compatibility between themselves and the clothes they want to wear.

The goal is to provide users with additional confidence when making fashion choices.

The application takes **a photo of clothing** as input.
By analyzing the color and silhouette of the clothing item, the system estimates how well it may match the user's personal color characteristics and body type.

---

# Features

## Color Analysis

* Average color extraction from clothing images
* Feature generation using color spaces (L*a*b and HSV)
* Personal color classification into **four seasonal types**
* Machine learning model using **RandomForest**

### Output

* Estimated personal color season compatibility
* Visualization of extracted color features

---

## Shape Analysis

* Image preprocessing
* Contour extraction using **OpenCV**
* Clothing silhouette classification using **YOLOv8**
* Rule-based body type scoring

### Output

* Predicted silhouette category
* Body type scoring results

---

# Explainable AI

To improve interpretability, the application visualizes prediction results.

* Visualization using **Altair**
* Charts showing analysis results
* Interpretable output for users

---

# Tech Stack

* Python
* Streamlit
* OpenCV
* scikit-learn (RandomForest)
* YOLOv8
* Altair
* NumPy
* Pandas

---

# Architecture

The system processes clothing images through two independent pipelines.

## Color Analysis Pipeline

1. Image upload via Streamlit
2. Average color extraction
3. Feature generation (L*a*b, HSV)
4. Personal color classification using RandomForest
5. Visualization of prediction results

## Shape Analysis Pipeline

1. Image preprocessing
2. Contour extraction using OpenCV
3. Clothing silhouette classification using YOLOv8
4. Rule-based body type scoring
5. Visualization of results

---

# Project Structure

fashion_app

├ fashion_app.py                 # Streamlit application
├ season_inference.py            # Personal color inference
├ silhouette_inference_local.py  # Silhouette classification
├ extract_colors_v2.py           # Color feature extraction
├ extract_shape_features_from_image.py

├ season_model_rf_proba.pkl      # Trained RandomForest model
├ best.pt                        # YOLOv8 classification model

├ train_season_rf_proba.py       # Model training script
├ score_body_type.py             # Body type scoring logic

└ requirements.txt

---

# How to Run Locally

Install dependencies

pip install -r requirements.txt

Run the Streamlit application

streamlit run fashion_app.py

---

# Future Improvements

* Improve silhouette classification accuracy
* Expand the training dataset
* Deploy the application as a web service
* Improve explainability features

---

# Author

This project was developed as part of an AI / Data Science portfolio.
