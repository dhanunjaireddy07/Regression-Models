# Regression-Models

🏠 Boston House Price Prediction

This project predicts housing prices using the Boston Housing Dataset with multiple regression techniques. It demonstrates how different models perform on the same dataset, including Linear Regression, Polynomial Regression, Lasso Regression, and Ridge Regression.

📌 Project Overview

The goal of this project is to compare different regression techniques on the Boston housing dataset. Each model is trained, tested, and evaluated based on:

-> Training Performance (R² Score)

-> Testing Performance (R² Score)

-> Mean Squared Error (MSE)

The models included:

-> Linear Regression

-> Polynomial Regression (degree=2)

-> Lasso Regression (α=0.01)

-> Ridge Regression (α=0.01)

📂 Project Structure
Boston-House-Prediction/
│-- housing_prediction.py   # Main script for training and evaluating models
│-- housing.csv             # Dataset (Boston Housing data)
│-- README.md               # Project documentation

⚙️ Requirements

Install dependencies using:

-> pip install numpy pandas matplotlib scikit-learn

📊 Example Output

The script will print:

-> Training and testing scores for each model

-> Mean squared error for each model

-> A summary table comparing all models

--> Example comparison table:

| Model                 | Training Performance | Testing Performance | Mean Squared Error |
|------------------------|----------------------|---------------------|---------------------|
| Linear Regression      | 0.750886             | 0.668750            | 24.291119          |
| Polynomial Regression  | 0.940932             | 0.805583            | 14.257338          |
| Lasso Regression       | 0.923544             | 0.785725            | 15.713579          |
| Ridge Regression       | 0.939936             | 0.819166            | 13.261267          |


(Values may vary depending on dataset split.)

📈 Key Learnings

-> Linear models provide a baseline.

-> Polynomial regression can improve accuracy but risks overfitting.

-> Lasso reduces overfitting using L1 regularization.

-> Ridge reduces overfitting using L2 regularization.

