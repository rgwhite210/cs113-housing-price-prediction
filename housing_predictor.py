"""
Housing Price Prediction with Linear Regression
================================================
This program loads housing data from a CSV file, visualizes the data,
trains a linear regression model, and predicts housing prices based on
user input.
Author: Rachel White
GitHub Link: https://github.com/rgwhite210/cs113-housing-price-prediction
Date: 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


def load_data(filepath):
    """
    Loads housing data from a CSV file.
    Returns a pandas DataFrame or None if loading fails.
    """
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully!")
        print(f"   Rows loaded: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return None


# --- Main ---
if __name__ == "__main__":
    df = load_data("housing_data.csv")
    print(df)