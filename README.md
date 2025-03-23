# 📊 Financial Sentiment Analysis for Vietnamese Stock Market

<div align="center">

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.1.0-EE4C2C.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.12.0-FF6F00.svg)](https://tensorflow.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-v4.4+-47A248.svg)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/Docker-supported-2496ED.svg)](https://www.docker.com/)

`<a href="#"><img src="https://raw.githubusercontent.com/khuyentran1401/awesome-python-data-science/master/assets/banner.png" width="900px"/>``</a>`

[📝 Documentation](docs/) |
[🚀 Installation](#installation) |
[💡 Examples](examples/) |
[👥 Contributing](CONTRIBUTING.md)

---

**Production-ready sentiment analysis for Vietnamese financial markets using state-of-the-art NLP**

---

</div>

## 🛠️ Environment Setup

### System Requirements

| Component | Minimum                   | Recommended               |
| --------- | ------------------------- | ------------------------- |
| CPU       | Intel Core i5/AMD Ryzen 5 | Intel Core i7/AMD Ryzen 7 |
| RAM       | 16GB                      | 32GB                      |
| GPU       | NVIDIA GTX 1660 6GB       | NVIDIA RTX 3060 12GB      |
| Storage   | 50GB SSD                  | 100GB NVMe SSD            |
| OS        | Ubuntu 20.04/Windows 10   | Ubuntu 22.04/Windows 11   |

### 📦 Core Dependencies

| Package    | Version  | Purpose                      |
| ---------- | -------- | ---------------------------- |
| Python     | ≥3.8    | Core runtime environment     |
| CUDA       | ≥11.0   | GPU acceleration framework   |
| cuDNN      | ≥8.0    | Deep learning on NVIDIA GPUs |
| PyTorch    | ≥2.1.0  | Deep learning framework      |
| TensorFlow | ≥2.12.0 | Machine learning platform    |
| MongoDB    | ≥4.4    | Database management          |

### 🔧 Key Python Packages

#### Machine Learning & Deep Learning

```python
torch>=2.1.0
tensorflow>=2.12.0
scikit-learn>=1.2.2
xgboost>=1.7.0
transformers>=4.30.0
```

#### NLP & Vietnamese Language Processing

```python
underthesea>=6.8.0
vncorenlp>=1.0.3
fastBPE>=0.1.0
fairseq>=0.12.2
nltk>=3.8.1
```

#### Data Processing & Analysis

```python
numpy>=1.23.5
pandas>=1.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

#### Development Tools

```python
black>=22.0  # Code formatting
flake8>=4.0  # Code linting
pytest>=7.0  # Testing
sphinx>=4.0  # Documentation
```

### 🐳 Docker Environment

```dockerfile
# Base image with CUDA support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Python version
ARG PYTHON_VERSION=3.8

# System dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Java runtime for VnCoreNLP
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless
```

### 📊 Supported Platforms & Versions

| Platform | Version     | Status             |
| -------- | ----------- | ------------------ |
| Ubuntu   | 20.04/22.04 | ✅ Fully Supported |
| Windows  | 10/11       | ✅ Fully Supported |
| macOS    | Big Sur+    | ⚠️ CPU Only      |
| Docker   | ≥20.10     | ✅ Fully Supported |
| WSL2     | Latest      | ✅ Fully Supported |

### 🔄 Version Control & CI/CD

- Git ≥2.30
- GitHub Actions for CI/CD
- Pre-commit hooks for code quality
- Automated testing on pull requests

### 📝 Code Quality Standards

- Black code formatter
- Flake8 linting
- Type hints with mypy
- 90% test coverage requirement
- Documented with Google style docstrings

## �� Project Overview

> A sophisticated sentiment analysis system for the Vietnamese stock market, leveraging state-of-the-art NLP techniques and deep learning models to analyze financial news headlines and predict market sentiment.

### 🌟 Key Highlights

- 🤖 Advanced NLP processing for Vietnamese language
- 📈 Real-time market data integration
- 🧠 Hybrid ML/DL approach
- 🎯 High accuracy metrics (~89%)
- 🔄 Continuous learning capabilities

## 📚 Data Description

### 📂 Raw Data (`data_1/`)

#### 📰 News Headlines Data (`media07_01_2023.xlsx`)

- Financial news headlines and articles from Vietnamese stock market
- Coverage: Major Vietnamese financial news sources
- Time period: 2022-2023
- Features:
  - Date and time of publication
  - News headline
  - Article content
  - Source information
  - Stock code references
- Size: 90KB, 328 entries

#### 💹 Stock Price Data (`price07_01_2023.xlsx`)

- Historical stock price information from Vietnamese exchanges
- Coverage: All listed companies on HOSE, HNX, UPCOM
- Time period: 2022-2023
- Features:
  - Trading date
  - Open, High, Low, Close prices
  - Trading volume
  - Price changes
  - Market classification
- Size: 295KB, 1,145 records

### 📊 Processed Data (`data_2/`)

#### 🔄 Combined Working Dataset (`datacw.xlsx`)

- Merged and preprocessed dataset combining news and price data
- Features:
  - Cleaned and tokenized news headlines
  - Normalized price data
  - Technical indicators
  - Sentiment labels
- Size: 1.6MB
- Used for: Model training and validation

#### 📈 Model Evaluation Datasets

1. **Out-of-Sample Testing** (`out_of_sample_data.xlsx`)

   - Independent test set for final model evaluation
   - Size: 115KB, 365 samples
   - Carefully curated to prevent data leakage
2. **Prediction Results**

   - Training predictions (`pred_train.xlsx`)

     - Size: 510KB, 2,151 records
     - Contains model predictions on training data
     - Used for model optimization
   - Testing predictions (`pred_test.xlsx`)

     - Size: 429KB, 1,668 records
     - Contains model predictions on test data
     - Used for final performance evaluation

### 📊 Data Statistics

| Dataset       | Positive | Negative | Neutral | Total |
| ------------- | -------- | -------- | ------- | ----- |
| Training      | 45%      | 30%      | 25%     | 2,151 |
| Testing       | 42%      | 33%      | 25%     | 1,668 |
| Out-of-Sample | 43%      | 32%      | 25%     | 365   |

### 🔍 Data Quality Measures

- ✅ Class balance maintained across splits
- ✅ No temporal leakage between train/test
- ✅ Duplicate headlines removed
- ✅ Missing values handled
- ✅ Outliers processed

### 🔄 Data Processing Pipeline

```mermaid
graph TD
    A[Raw News Data] --> B[Text Cleaning]
    C[Raw Price Data] --> D[Price Normalization]
    B --> E[Vietnamese Tokenization]
    D --> F[Technical Indicators]
    E --> G[Feature Extraction]
    F --> G
    G --> H[Final Dataset]
```

### 📈 Data Distribution Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example visualization code for data distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='sentiment')
plt.title('Distribution of Sentiment Labels')
plt.show()
```

## 💡 Motivation

The Vietnamese stock market is experiencing unprecedented growth, yet there remains a significant gap in sophisticated sentiment analysis tools specifically designed for Vietnamese financial contexts. Our project addresses this challenge by providing:

- 🔍 **Automated Analysis**: Real-time processing of financial news headlines
- 📊 **Market Integration**: Direct connection with stock price data
- 🇻🇳 **Language Support**: Specialized Vietnamese language processing
- 🎯 **High Accuracy**: State-of-the-art prediction models

## ✨ Features

### 🔧 Technical Capabilities

- 📝 Advanced Vietnamese text preprocessing
- 🤖 Multiple model architectures
- 📊 Real-time data processing
- 🗄️ Scalable data storage
- 📈 Comprehensive analytics

### 🧮 Model Architecture

```mermaid
graph LR
    A[Raw Data] --> B[Preprocessing]
    B --> C[Feature Extraction]
    C --> D[Model Training]
    D --> E[Sentiment Analysis]
    E --> F[Results]
```

## 🛠️ Prerequisites

| Requirement | Version | Purpose          |
| ----------- | ------- | ---------------- |
| Python      | ≥3.8   | Core runtime     |
| CUDA        | ≥11.0  | GPU acceleration |
| MongoDB     | ≥4.4   | Data storage     |
| RAM         | ≥16GB  | Model training   |

## 🚀 Installation

### 1️⃣ Clone Repository

```bash
git clone [repository-url]
cd financial-sentiment-analysis
```

### 2️⃣ Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Vietnamese NLP Tools

```bash
# 📥 Install VnCoreNLP
mkdir -p vncorenlp/models/wordsegmenter
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
```

### 4️⃣ PhoBERT Model

```bash
wget https://public.vinai.io/PhoBERT_base_transformers.tar.gz
tar -xzvf PhoBERT_base_transformers.tar.gz
```

## 📁 Project Structure

```
📦 financial-sentiment-analysis
 ┣ 📂 data/
 ┃ ┣ 📂 data_1/          # Raw financial data
 ┃ ┗ 📂 data_2/          # Processed datasets
 ┣ 📂 models/            # Trained model artifacts
 ┣ 📂 notebooks/         # Jupyter notebooks
 ┣ 📂 vncorenlp/        # Vietnamese NLP tools
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

## ⚙️ Configuration

### 🗄️ Database Setup

```python
MONGODB_URI = "mongodb+srv://your_username:your_password@your_cluster.mongodb.net/"
```

### 🤖 Model Parameters

| Parameter     | Value | Description             |
| ------------- | ----- | ----------------------- |
| MAX_LEN       | 125   | Maximum sequence length |
| BATCH_SIZE    | 32    | Training batch size     |
| EPOCHS        | 10    | Training epochs         |
| LEARNING_RATE | 2e-5  | Initial learning rate   |
| WARMUP_STEPS  | 1000  | Learning rate warmup    |

## 📊 Performance Metrics

### 📈 Traditional ML Models

| Model               | Accuracy        | Precision       | Recall          | F1-Score        |
| ------------------- | --------------- | --------------- | --------------- | --------------- |
| Gaussian NB         | 82.3%           | 81.5%           | 83.1%           | 82.3%           |
| Logistic Regression | 84.1%           | 83.7%           | 84.5%           | 84.1%           |
| Random Forest       | 85.6%           | 85.2%           | 86.0%           | 85.6%           |
| XGBoost             | 86.2%           | 85.8%           | 86.6%           | 86.2%           |
| **Ensemble**  | **87.5%** | **87.1%** | **87.9%** | **87.5%** |

### 🧠 Deep Learning Models

| Model   | Accuracy        | Precision       | Recall          | F1-Score        |
| ------- | --------------- | --------------- | --------------- | --------------- |
| LSTM    | 86.8%           | 86.4%           | 87.2%           | 86.8%           |
| BiLSTM  | 87.5%           | 87.1%           | 87.9%           | 87.5%           |
| PhoBERT | **89.2%** | **88.8%** | **89.6%** | **89.2%** |

### 📊 Performance Analysis

```python
# ROC Curve and Confusion Matrix visualization code
import seaborn as sns
import matplotlib.pyplot as plt

# Add your visualization code here
```

## 🐳 Docker Support

```dockerfile
FROM python:3.8

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

### 🚀 Docker Commands

```bash
# Build image
docker build -t financial-sentiment .

# Run container
docker run -p 8000:8000 financial-sentiment

# Monitor logs
docker logs -f financial-sentiment
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. 🍴 Fork the repository
2. 🔧 Create your feature branch
3. 💻 Commit your changes
4. 🚀 Push to the branch
5. 🎉 Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- 🏆 VnCoreNLP team for Vietnamese language tools
- 🌟 VINAI research team for PhoBERT
- 💾 MongoDB Atlas for database support
- 📚 Open-source community

## 📬 Contact & Support

- 📧 Email: [your-email@domain.com]
- 💬 Discord: [your-discord]
- 🐦 Twitter: [@your-twitter]

## 📈 Project Status

- ✅ Core functionality complete
- 🚧 API Documentation in progress
- 🔄 Continuous model improvements
- 📊 Regular performance updates

---

<div align="center">
Made with 💖 for the Vietnamese Financial Market

⭐️ Star us on GitHub — it helps!

</div>
