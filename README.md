# Hybrid Movie Recommendation System 🎬

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)

A sophisticated hybrid recommendation engine combining content-based and collaborative filtering techniques using the MovieLens 100K dataset.

![System Demo](demo.gif) *Replace with actual demo gif*

## 🌟 Features

- **Dual Recommendation Approach**:
  - 🎭 Content-based filtering using TF-IDF and cosine similarity
  - 👥 Collaborative filtering using SVD matrix factorization
- **Intelligent Hybridization**:
  - ⚖️ Weighted combination of both methods
  - 🎚️ Adjustable weights for optimal recommendations
- **User-Friendly Interface**:
  - 🖥️ Streamlit web application
  - 🔍 Both movie-based and user-based recommendation modes
- **Comprehensive Evaluation**:
  - 📊 RMSE and MAE performance metrics
  - 🎯 Recommendation diversity analysis

## 🏗️ System Architecture
 ┌───────────────────────┐
│ MovieLens 100K │
│ Dataset │
└──────────┬───────────┘
│
┌──────────▼───────────┐
│ Data Processing │
└──────────┬───────────┘
│
┌──────────▼───────────┐ ┌───────────────────────┐
│ Content-Based │ │ Collaborative │
│ Recommender │ │ Recommender │
└──────────┬───────────┘ └──────────┬────────────┘
│ │
└──────────┬───────────────┘
│
┌──────────▼───────────┐
│ Hybrid Engine │
└──────────┬───────────┘
│
┌──────────▼───────────┐
│ Streamlit UI │
└──────────────────────┘


## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/hybrid-movie-recommender.git
cd hybrid-movie-recommender

nstall dependencies:

bash
pip install -r requirements.txt
Run the application:

bash
streamlit run app.py
📂 Project Structure
hybrid-recommender/
├── data_processing.py    # Data loading and preprocessing
├── content_based.py      # Content-based filtering implementation
├── collaborative.py      # Collaborative filtering implementation
├── hybrid.py             # Hybrid recommendation engine
├── evaluation.py         # Model evaluation metrics
├── app.py                # Streamlit web application
├── requirements.txt      # Dependencies
└── README.md             # This file
📊 Evaluation Results
Model	RMSE	MAE
Collaborative Filtering	0.932	0.734
Hybrid Approach	0.915	0.712
Recommendation Diversity

🌐 Deployment Options
Option 1: Streamlit Sharing
Push your code to GitHub

Go to Streamlit Sharing

Connect your GitHub account and deploy

Option 2: Docker
bash
docker build -t movie-recommender .
docker run -p 8501:8501 movie-recommender
Option 3: AWS Elastic Beanstalk
bash
eb init -p python-3.8 movie-recommender
eb create movie-recommender-env
