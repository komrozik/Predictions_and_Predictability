import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from sklearn.linear_model import LinearRegression


def climatology_pred(df, start_year, end_year, window_length=30, variable="Tropical Storms"):
    """
    Predicts climatological trends for a specified variable over a range of years within a DataFrame.

    Parameters:
    - df (DataFrame): Input DataFrame containing climatological data with a "Year" column and the specified variable.
    - start_year (int): Starting year for the analysis.
    - end_year (int): Ending year for the analysis.
    - window_length (int, optional): Length of the window used for calculating climatological trends. Default is 30.
    - variable (str, optional): The column name in the DataFrame representing the variable for prediction. Default is "Tropical Storms".

    Returns:
    - predictions (list): List of predicted values for the specified variable within each window.
    - start_years (list): List of starting years for each window.
    - end_years (list): List of ending years for each window.
    """

    # Initialize empty lists to store predictions, start years, and end years
    predictions = []
    start_years = []
    end_years = []

    # Loop through the specified range of years for creating sliding windows
    for year in range(start_year, end_year - (window_length - 1)):
        # Create a window of data within the specified range
        window = df[(df["Year"] >= year) & (df["Year"] < year + window_length)]

        # Calculate the mean value of the specified variable within the window
        prediction = window[variable].mean()

        # Append prediction, start year, and end year to their respective lists
        predictions.append(prediction)
        start_years.append(year)
        end_years.append(year + (window_length - 1))
    
    # Return the lists of predictions, start years, and end years as a tuple
    return (predictions, start_years, end_years)



def linear_reg_pred(df, start_year, end_year, window_length=30, variable="Tropical Storms"):
    """
    Predicts future values using linear regression for a specified variable over a range of years within a DataFrame.

    Parameters:
    - df (DataFrame): Input DataFrame containing data with a "Year" column and the specified variable.
    - start_year (int): Starting year for the analysis.
    - end_year (int): Ending year for the analysis.
    - window_length (int, optional): Length of the window used for linear regression. Default is 30.
    - variable (str, optional): The column name in the DataFrame representing the variable for prediction. Default is "Tropical Storms".

    Returns:
    - predictions (list): List of predicted values using linear regression for the specified variable within each window.
    - start_years (list): List of starting years for each window.
    """

    # Initialize empty lists to store predictions and start years
    predictions = []
    start_years = []

    # Loop through the specified range of years for creating sliding windows
    for year in range(start_year, end_year - (window_length - 1)):
        # Create a window of data within the specified range
        window = df[(df["Year"] >= year) & (df["Year"] < year + window_length)]

        # Extract x (Year) and y (specified variable) values for linear regression
        x = window["Year"].to_numpy().reshape((-1, 1))
        y = window[variable].to_numpy()

        # Fit a linear regression model to the window data
        model = LinearRegression().fit(x, y)

        # Calculate coefficient of determination (R-squared)
        r_sq = model.score(x, y)

        # Extract the intercept and slope of the linear regression line
        intercept = model.intercept_
        slope = model.coef_

        # Predict future values based on the linear regression equation
        prediction = f(m=slope, b=intercept, start=year, end=2025)  # Custom function f() for prediction
        predictions.append(prediction)
        start_years.append(year)
    
    # Return the lists of predictions and start years as a tuple
    return (predictions, start_years)

def persistence_pred(df, start_year, end_year, window_length=5, variable="Tropical Storms"):
    """
    Predicts future values using persistence (average of past values) for a specified variable over a range of years within a DataFrame.

    Parameters:
    - df (DataFrame): Input DataFrame containing data with a "Year" column and the specified variable.
    - start_year (int): Starting year for the analysis.
    - end_year (int): Ending year for the analysis.
    - window_length (int, optional): Length of the window used for persistence prediction. Default is 5.
    - variable (str, optional): The column name in the DataFrame representing the variable for prediction. Default is "Tropical Storms".

    Returns:
    - predictions (list): List of predicted values using persistence for the specified variable.
    - years (list): List of years for which predictions were made.
    """

    # Initialize empty lists to store predictions and years
    predictions = []
    years = []

    # Loop through the specified range of years for creating persistence predictions
    for year in range(start_year, end_year):
        # Create a window of data within the specified range for persistence calculation
        window = df[(df["Year"] >= year - window_length) & (df["Year"] < year)]

        # Calculate persistence prediction (average of past values)
        pred = window[variable].mean()
        predictions.append(pred)
        years.append(year)
    
    # Return the lists of predictions and years as a tuple
    return (predictions, years)

