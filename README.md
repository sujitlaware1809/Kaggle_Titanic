# Titanic - Machine Learning from Disaster

This is a beginner-friendly machine learning project based on the classic **Titanic dataset** from [Kaggle](https://www.kaggle.com/competitions/titanic).

## Goal

Predict whether a passenger survived or not using features like age, gender, ticket class, etc.

---

## Files

- `train.csv` - Training data with survival labels  
- `test.csv` - Test data without survival labels  
- `submission.csv` - Final predictions to upload on Kaggle  
- `titanic_model.ipynb` - Jupyter notebook with code

---

## What I Did

1. Loaded and explored the dataset
2. Cleaned data (handled missing values)
3. Converted categorical data to numeric
4. Trained a machine learning model
5. Made predictions on test data
6. Saved predictions to `submission.csv`

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
