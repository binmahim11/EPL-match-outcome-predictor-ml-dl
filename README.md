# EPL-match-outcome-predictor-ml-dl

This project predicts English Premier League (EPL) match results using Python, machine learning, and deep learning models. It analyzes past match data and team statistics to predict **win**, **loss**, or **draw** outcomes.
## 📌 Overview

The goal of this project is to build an accurate model that forecasts EPL match results by learning patterns from historical match data. Both traditional ML algorithms and Deep Learning approaches are used.


## 🚀 Features

* Data cleaning and preprocessing
* Feature engineering from match statistics
* ML models: Logistic Regression, Random Forest, XGBoost
* Deep Learning model (ANN/LSTM)
* Model comparison and evaluation (accuracy, F1-score)
* Exported trained model for predictions


## 🧠 Tech Stack

* Python
* Pandas, NumPy
* Scikit-Learn
* TensorFlow / Keras
* Matplotlib, Seaborn
* XGBoost / LightGBM


## 📊 Dataset

The dataset includes historical EPL match data with:

* Team stats
* Goals scored & conceded
* Shots, xG, possession
* Home/away performance
* Match results (Win/Draw/Loss)


## 📂 Project Structure

```
├── app.py              # Streamlit main application
├── data/
│   └── epl_matches.csv
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   └── final_model.pkl
├── requirements.txt
└── README.md
```

├── data/
│   └── epl_matches.csv
├── notebooks/
│   └── model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_ml_models.py
│   ├── train_dl_model.py
│   └── predict.py
├── models/
│   └── final_model.pkl
└── README.md



## ▶️ How to Run
1. Clone the repository:
```

git clone <repo-url>

```
2. Install dependencies:
```

pip install -r requirements.txt

```
3. Run the model training notebook:
```

notebooks/model_training.ipynb



## 🎯 Goal
To build a reliable sports prediction system and provide a ready-to-use model for EPL match forecasting.


```

