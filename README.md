# Machine Learning  Course with Python

A focused project repository covering **supervised learning regression** techniques using Python. This course emphasizes practical implementation of multiple regression algorithms on real-world datasets, providing hands-on experience with data preprocessing, model training, evaluation, and visualization.

---

## Introduction

Supervised learning is a fundamental machine learning paradigm where models learn to predict continuous target values from input features. Regression, a key supervised learning task, estimates relationships between dependent and independent variables.

This repository contains implementations of various **regression algorithms**, including:

- Linear Regression (Simple & Multiple)
- Logistic Regression (for classification tasks with regression concepts)
- Support Vector Regression (SVR)
- K-Nearest Neighbors Regression (KNN)
- Gamma Regression
- Spline Regression

Each algorithm is demonstrated with clean, modular Python code, supported by detailed visualizations and performance metrics.

---

## Theory Overview

### What is Regression?

Regression analysis predicts a continuous outcome variable based on one or more predictor variables. It helps uncover trends, dependencies, and forecast future values.

### Types of Regression Covered

- **Linear Regression**: Models the linear relationship between features and target.
- **Multiple Linear Regression**: Extension of linear regression with multiple predictors.
- **Logistic Regression**: Although a classification method, it uses a regression framework to model probabilities.
- **Support Vector Regression (SVR)**: Applies the principles of Support Vector Machines to regression tasks, handling non-linear relationships with kernel functions.
- **K-Nearest Neighbors Regression (KNN)**: Predicts target values by averaging the nearest neighbors in the feature space.
- **Gamma Regression**: A generalized linear model suitable for positive continuous data, modeling data with Gamma distribution.
- **Spline Regression**: Uses piecewise polynomials for flexible, smooth curve fitting.

---
###  Prerequisites  

Install the required libraries:
## Required Python Packages

This project requires the following Python packages for data analysis, visualization, and implementing supervised learning regression algorithms:

- **numpy** — Numerical computing library for handling arrays and mathematical functions.  
- **ipython** — Interactive Python shell for enhanced coding experience.  
- **jupyter** — Run Jupyter Notebooks interactively for exploratory programming and visualization.  
- **pandas** — Data manipulation and analysis tool, perfect for working with structured datasets.  
- **matplotlib** — Core plotting library used to create static, animated, and interactive visualizations.  
- **seaborn** — Statistical data visualization built on top of matplotlib with attractive default styles.  
- **scikit-learn** — Robust machine learning library offering regression algorithms such as Linear Regression, Support Vector Regression (SVR), K-Nearest Neighbors (KNN), Logistic Regression, and tools for hyperparameter tuning.  
- **scipy** — Scientific computing library used for advanced regression types like Gamma and Spline Regression.  
- **statsmodels** — Advanced statistical modeling and diagnostic tools, highly useful for regression analysis and spline regression implementations.

---

### Install all dependencies with:

```bash
pip install numpy ipython jupyter pandas matplotlib seaborn scikit-learn scipy statsmodels


** Run the Notebooks**
jupyter notebook


 **📁 Project Structure **
machine-learning-regression-python/
│
├── supervised-learning/
│   ├── regression/
│   │   ├── linear_regression/
│   │   ├── multiple_linear_regression/
│   │   ├── logistic_regression/
│   │   ├── support_vector_regression/
│   │   ├── gamma_regression/
│   │   ├── spline_regression/
│   │   └── knn_regression/
│
├── README.md
└── LICENSE

🎯 Learning Outcomes
Understand the fundamentals and mathematics behind popular regression models

Implement linear, logistic, support vector, gamma, spline, and KNN regression from scratch and with libraries

Perform data preprocessing, feature scaling, and model evaluation

Apply hyperparameter tuning for better model performance

Visualize model results with plots including residuals, learning curves, and feature importance

🔑 Keywords
Regression, Machine Learning, Supervised Learning, Linear Regression, Multiple Linear Regression, Logistic Regression, Support Vector Regression, SVR, Gamma Regression, Spline Regression, KNN Regression, K-Nearest Neighbors, Python, scikit-learn, numpy, pandas, matplotlib, seaborn, data science, predictive modeling, hyperparameter tuning

** Contributing**
Feel free to fork this repo and submit a pull request if you'd like to add:

More ML algorithms

Advanced DL architectures

Project examples

⭐ If you found this helpful...
Please consider giving the repo a ⭐ — it helps others discover the project!
