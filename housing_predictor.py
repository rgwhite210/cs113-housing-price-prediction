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

def explore_data(df):
    """
    Displays basic statistics and a scatter plot of the housing data.
    """
    try:
        print("\nBasic Statistics:")
        print("-" * 40)
        print(f"  Count : {len(df)}")
        print(f"  [Size] Min: {df['Size (sqft)'].min():,} | Max: {df['Size (sqft)'].max():,} | Mean: {df['Size (sqft)'].mean():,.0f}")
        print(f"  [Price] Min: ${df['Price ($)'].min():,} | Max: ${df['Price ($)'].max():,} | Mean: ${df['Price ($)'].mean():,.0f}")

        # Scatter plot
        plt.figure(figsize=(8, 5))
        plt.scatter(df['Size (sqft)'], df['Price ($)'], color='steelblue', alpha=0.7, edgecolors='black')
        plt.title('House Size vs Price')
        plt.xlabel('Size (sqft)')
        plt.ylabel('Price ($)')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error exploring data: {e}")

# Main
if __name__ == "__main__":
    df = load_data("housing_data.csv")
    if df is not None:
        explore_data(df)