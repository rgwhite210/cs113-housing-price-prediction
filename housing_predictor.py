"""
Housing Price Prediction with Linear Regression
This program loads housing data from a CSV file, visualizes the data,
trains a linear regression model, and predicts housing prices based on
user input.
Author: Rachel White
GitHub Link: https://github.com/rgwhite210/cs113-housing-price-prediction
Date: March 2026
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

def train_model(df):
    """
    Trains a linear regression model on the housing data.
    Returns the trained model and the feature/target arrays.
    """
    try:
        # Prepare the data
        X = df[['Size (sqft)']].values  # Feature (2D array required by sklearn)
        y = df['Price ($)'].values       # Target

        # Train the model
        model = LinearRegression()
        model.fit(X, y)

        # Calculate R² score
        y_predicted = model.predict(X)
        r2 = r2_score(y, y_predicted)

        # Display model details
        print("\nModel Training Complete!")
        print("-" * 40)
        print(f"  Coefficient  : ${model.coef_[0]:,.2f} per sqft")
        print(f"  Intercept    : ${model.intercept_:,.2f}")
        print(f"  R² Score     : {r2:.4f}")
        print("\n  Interpretation:")
        print(f"  → For every 1 sqft increase, price increases by ${model.coef_[0]:,.2f}")
        print(f"  → R² of {r2:.4f} means the model explains {r2*100:.1f}% of price variation")

        return model, X, y

    except Exception as e:
        print(f"Error training model: {e}")
        return None, None, None

def visualize_model(X, y, model):
    """
    Displays a scatter plot of the data points along with the regression line.
    """
    try:
        # Generate the regression line
        regression_line = model.predict(X)

        # Plot
        plt.figure(figsize=(8, 5))

        # Scatter plot of actual data points
        plt.scatter(X, y, color='steelblue', alpha=0.7, edgecolors='black', label='Actual Prices')

        # Regression line
        plt.plot(X, regression_line, color='red', linewidth=2, label='Regression Line')

        plt.title('House Size vs Price with Regression Line')
        plt.xlabel('Size (sqft)')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error visualizing model: {e}")

# Main
if __name__ == "__main__":
    df = load_data("housing_data.csv")
    if df is not None:
        explore_data(df)
        model, X, y = train_model(df)
        if model is not None:
            visualize_model(X, y, model)