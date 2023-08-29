# Customer Lifetime Value Prediction Project

## Overview

This project aims to predict Customer Lifetime Value (CLV) for a business using Python. CLV is a critical metric that helps businesses understand the long-term value of their customers, enabling better decision-making in marketing, sales, and customer retention strategies.

## Project Contents

### Data Preparation
The project starts by loading historical customer transaction data and performing data preprocessing to calculate Recency, Frequency, and Monetary Value (RFM).

### Visualization
Visualizations, created with `matplotlib`, are included to explore the relationships between Recency, Frequency, and Log(Monetary Value). A histogram of Log(Monetary Value) helps understand its distribution.

### CLV Prediction Model
A Linear Regression model, implemented with scikit-learn, is used to predict CLV based on RFM metrics.

### Model Evaluation
The model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R2) metrics. A scatter plot compares actual vs. predicted CLV values.

### Usage
Detailed instructions on how to use the code to predict CLV for specific customers.

## Dependencies


Now, the author's name, Harshit Vinayak, is included in the README.


Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy matplotlib scikit-learn
