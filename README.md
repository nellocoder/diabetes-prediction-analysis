# diabetes-prediction-analysis
End-to-end analysis and prediction of diabetes risk using the Pima Indians dataset.
# Metabolic Health Explorer: Pima Indians Diabetes Analysis

## Project Overview
This project analyzes the Pima Indians Diabetes Dataset to identify key indicators of diabetes risk. It progresses from Exploratory Data Analysis (EDA) to building predictive models using Python.

## Key Insights
* **Data Integrity:** Identified that the dataset used `0` as a placeholder for missing values in biologically impossible columns (Glucose, Blood Pressure, BMI).
* **Correlation:** Found strong correlations between Glucose levels and Diabetes outcome.
* **Outliers:** Handled missing data by imputing median values to preserve distribution integrity.

## Tech Stack
* **Python:** Pandas for data manipulation.
* **Visualization:** Seaborn & Matplotlib for heatmaps and distribution plots.
* **Machine Learning:** Scikit-Learn (Linear Regression, Random Forest, Logistic Regression).

## How to Run
1.  Clone the repository.
2.  Install dependencies: `pip install pandas seaborn scikit-learn`.
3.  Run the script `metabolic_health_eda.py`.
## Model Performance (Logistic Regression)
We trained a Logistic Regression model to predict diabetes onset based on diagnostic measures.

* **Overall Accuracy:** 75.32%
* **Key Insight:** The model is highly effective at identifying healthy patients (Precision: 0.80) but requires further tuning to improve sensitivity for detecting positive diabetic cases (Recall: 0.62).

### Confusion Matrix Results
| | Predicted Healthy | Predicted Diabetic |
|---|---|---|
| **Actual Healthy** | High Accuracy | Low False Positives |
| **Actual Diabetic** | Moderate False Negatives | Moderate True Positives |
