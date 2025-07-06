# Loan Default Prediction using Neural Networks

This project aims to predict loan defaults based on historical LendingClub data using an artificial neural network built with Keras.

## Objective

The goal is to support loan approval decisions by:
- Minimizing financial losses from high-risk loans.
- Maximizing business opportunities by identifying trustworthy applicants.

## Methodology

The project follows a full machine learning pipeline:

1. **Data Cleaning and Feature Engineering**
   - Handling missing values
   - Creating domain-specific features (e.g., `loan/income ratio`)
   - One-hot encoding and embedding layers for categorical variables

2. **Modeling**
   - Neural network built with Keras combining:
     - Numerical features
     - Categorical features via embeddings
   - Trained on a dataset of over 350,000 loans

3. **Threshold Optimization**
   - Custom classification threshold to maximize recall on the default class
   - Performance evaluated using recall, precision, and confusion matrix

4. **Model Explainability**
   - Feature importance assessed with SHAP (global and local explainability)

## Results

- Recall on the "default" class greater than 0.75
- Balanced precision/recall performance
- SHAP plots highlight the most influential features

## Project Structure

```
data/
├── lending_club_loan_two.csv        # Raw dataset
├── df_final.csv                     # Cleaned and transformed dataset

models/
├── loan_default_model.h5            # Trained Keras neural network

outputs/
├── predictions_with_threshold.csv   # Final predictions with adjusted threshold

Projet_publique_ANN.ipynb            # Main notebook
README.md                            # Project description
```


## Technologies

Python, scikit-learn, Keras, Keras Tuner, SHAP, pandas, matplotlib, Google Colab

## Getting Started

1. Clone the repository:
```
git clone https://github.com/lucwaghorn/lending_club.git
cd lending_club
```
