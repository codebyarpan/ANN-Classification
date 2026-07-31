# Customer Churn Prediction using Artificial Neural Network (ANN)

A deep learning project that predicts whether a bank customer is likely to churn based on demographic and account information. The project is built using TensorFlow/Keras and deployed with Streamlit.

## Features

- Customer churn prediction using ANN
- TensorFlow / Keras model
- Data preprocessing with Scikit-learn
- Interactive Streamlit web application
- Probability-based prediction

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit

## Dataset

The project uses the **Churn Modelling Dataset**.

Features include:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Active Member
- Estimated Salary

Target:

- **Exited**
  - 0 → Customer stays
  - 1 → Customer churns

## Project Structure

```
ANN-Classification/
│── app.py
│── experiments.ipynb
│── prediction.ipynb
│── model.keras
│── scaler.pkl
│── label_encoder_gender.pkl
│── onehot_encoder_geo.pkl
│── requirements.txt
│── README.md
│── .gitignore
```

## Installation

```bash
git clone git@github.com:codebyarpan/ANN-Classification.git
cd ANN-Classification

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

## Model

- Neural Network (ANN)
- Optimizer: Adam
- Loss: Binary Crossentropy
- Output Activation: Sigmoid

## Future Improvements

- Hyperparameter tuning
- Docker deployment
- Cloud deployment
- Explainable AI (SHAP/LIME)

## Author

**Arpan Pandey**

GitHub: https://github.com/codebyarpan