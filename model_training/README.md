# 🔒 Credit Fraud Detection Model

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A robust credit fraud detection system with comprehensive version management, environment control, and model registry operations. Built for scalability and production-ready deployment.

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Version Control](#-version-control)
- [Model Training](#-model-training)
- [Contributing](#-contributing)

## 🎯 Overview

This repository implements an advanced credit fraud detection pipeline using XGBoost and Scikit-Learn, featuring:
- Semantic versioning for model iterations
- Environment-specific deployment management
- Automated model registry operations
- Comprehensive changelog tracking

## ✨ Features

### 🏗️ Version Management
- **Semantic Versioning** (`MAJOR.MINOR.PATCH`)
  ```python
  version = ModelVersion(1, 0, 0, ModelEnvironment.PROD)
  ```
- **Environment Controls** (DEV/STAGING/PROD)
- **Automated Version Tracking**

### 📦 Model Registry
- Version-aware model storage
- Metadata management
- Historical tracking
- Environment isolation

### 🔄 Version Tracking
- Automatic version incrementation
- Detailed changelogs
- Training metadata preservation

## 📁 Project Structure

```
.
├── 📄 Dockerfile
├── 📝 README.md
├── 📁 data/
│   ├── application_test.csv
│   └── application_train.csv
├── 📁 models/
│   └── model.pkl
├── 📄 requirements.txt
└── 📁 src/
    ├── Loan_defaulters_prediction.ipynb
    ├── model_testing.py
    └── model_training.py
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required packages from `requirements.txt`

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/credit-fraud-detection.git
   cd credit-fraud-detection
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Training the Model

```bash
python src/model_training.py \
    --train-path data/application_train.csv \
    --test-path data/application_test.csv \
    --model-output models/credit_fraud_model.pkl \
    --environment dev
```

## 🔄 Version Control

### Version Format
```
v1.0.0-prod-20240401
│ │ │  │    │
│ │ │  │    └── Date
│ │ │  └─────── Environment
│ │ └────────── Patch
│ └──────────── Minor
└────────────── Major
```

### Environments
- 🔧 **DEV**: Development and experimentation
- 🧪 **STAGING**: Pre-production testing
- 🚀 **PROD**: Production deployment

## 📊 Model Training

### Training Process
1. Data preprocessing
2. Feature engineering
3. Model training
4. Evaluation
5. Version increment
6. Metadata storage

### Key Metrics
- 📈 Precision
- 📊 Recall
- 📉 ROC AUC Score

### Changelog Example
```json
{
    "version": "v1.0.0-prod-20240401",
    "changes": [
        {
            "type": "training",
            "description": "Initial model training",
            "timestamp": "2024-04-01T10:00:00Z"
        }
    ]
}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/amazing-feature
   ```
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
    Made with ❤️ for better credit fraud detection
</div>
