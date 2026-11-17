# Bike Rental Demand – Machine Learning

Predicting daily bike-rental demand for a bike-sharing company (Washington D.C., 2011–2012) with classic machine-learning models in **scikit-learn**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oded-tsfati/ML/blob/main/ML_Bike_Rental.ipynb)

## Goal
Use calendar and weather features (season, year, month, holiday, weekday, working day, weather situation) to:
1. **Regression** – predict `cnt`, the total number of rentals per day.
2. **Classification** – predict whether a day will be *busier than average* (more than ~4,500 rentals).

## Workflow
1. **EDA** – distributions, monthly patterns, missing-value checks
2. **Data preparation** – feature selection
3. **Modelling** – Linear Regression, Logistic Regression, Decision Tree
4. **Evaluation** – mean absolute error, accuracy, confusion matrix, predicted-vs-actual plots

## Results (test set, 25 % hold-out)
| Model | Task | Metric | Result |
|---|---|---|---|
| Linear Regression | daily rentals | Mean absolute error | ≈ 1,070 rentals/day |
| Logistic Regression | busy vs. quiet day | Accuracy | ≈ 69 % |

A Decision Tree model is also explored at the end of the notebook, including a visualisation of the tree.

**Takeaway:** season, year and weather explain a large part of demand – the second year shows clear growth and bad weather cuts rentals sharply.

## Data
`day.csv` – [Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), UCI Machine Learning Repository (731 days, 2011–2012), licensed CC BY 4.0.

## Tech stack
Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Google Colab

## How to run
```bash
pip install -r requirements.txt
jupyter notebook ML_Bike_Rental.ipynb
```
In the notebook, change the path in `pd.read_csv('/content/day.csv')` to `day.csv` when running locally.
