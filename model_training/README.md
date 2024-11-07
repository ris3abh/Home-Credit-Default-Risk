Credit Fraud Detection Model
This project implements a credit fraud detection model, designed to handle semantic versioning, environment management, and model registry operations. It provides enhanced version tracking, changelog management, and metadata storage, allowing for a structured and scalable model deployment.

Table of Contents
Project Overview
Features
- Version Management
- Model Registry
- Version Tracking
Project Structure
Getting Started
- Requirements
- Setup
- Usage
Version Control and Environments
Model Training and Evaluation
Changelog Tracking
Contributing
License
Project Overview

This repository contains a credit fraud detection model pipeline using XGBoost and Scikit-Learn. The pipeline includes features like version tracking, environment management, and automated model saving and loading for deployment and experimentation in different environments (DEV, STAGING, PROD).

Features
Version Management
The model uses semantic versioning `(MAJOR.MINOR.PATCH)` for systematic model updates, managed by the `ModelVersion` class.

`ModelVersion`: Tracks version metadata, environment, and timestamps.
`ModelEnvironment`: Manages DEV, STAGING, and PROD environments.
`Model Registry`: The ModelRegistry class is responsible for storing and retrieving models by version.

`Version-aware saving/loading`: Enables saving models with unique version identifiers.
`Metadata storage`: Keeps changelogs, training parameters, and environment information with each model.
`Version Tracking`: Includes automatic version increments after each training session and changelog tracking for easy reference.

Automatic Version Incrementation: Each training session updates the minor version.
Environment-specific Versioning: Maintains separate versions for each environment.


Project Structure
.
├── Dockerfile               # Dockerfile for containerization
├── README.md                # Project documentation
├── data                     # Directory for training and test datasets
│   ├── application_test.csv
│   └── application_train.csv
├── models                   # Directory for model storage
│   └── model.pkl
├── requirements.txt         # Python dependencies
└── src                      # Source code
    ├── Loan defaulters prediction.ipynb
    ├── model_testing.py     # Script for testing the model
    └── model_training.py    # Script for training the model

Getting Started

Requirements
- Python 3.8+
- Packages listed in requirements.txt
- Setup
- Clone the Repository

bash

`git clone https://github.com/yourusername/credit-fraud-detection.git`
`cd credit-fraud-detection`
`Install Dependencies`

`pip install -r requirements.txt`

Run Training Script

Use main function in model_training.py to train and evaluate the model with command-line arguments for paths and environment.

Usage

Running the Model Training

`python src/model_training.py --train-path data/application_train.csv --test-path data/application_test.csv --model-output models/credit_fraud_model.pkl --environment dev`
Version Control and Environments

The ModelEnvironment enum `(DEV, STAGING, PROD)` handles environment-specific settings. The ModelVersion class facilitates structured versioning by incrementing minor versions upon each training.

Example Version Format

v1.0.0-prod-20240401 where:

- 1.0.0 represents the MAJOR, MINOR, PATCH version
- prod is the environment
- 20240401 is the date the version was created

Model Training and Evaluation
- The CreditFraudModel class provides methods for training and evaluating the model. Each time the model is trained, the version can increment based on the environment.

Training: Trains the model using the provided dataset.
Evaluation: Evaluates the model and logs evaluation results, saving predictions to predictions.csv if a test dataset is provided.

Key Metrics
- Precision
- Recall
- ROC AUC Score

Changelog Tracking
Every model version includes a changelog that records changes, training sessions, and other relevant information. The changelog is stored as metadata with each model version.

Contributing
If you wish to contribute, please fork the repository and make a pull request. For major changes, open an issue first to discuss what you would like to change.

