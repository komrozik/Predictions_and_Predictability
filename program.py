import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression

import prediction_functions as pf

"""
This file will include all Plots we decide to finally use and will be updated by functions and stuff from the test_programming.ipynb.
"""

#-------------------------
# Load Data
#-------------------------

df = pd.read_csv("data/tcatlantic.csv")


#-------------------------
# Functions
#-------------------------

def acf(x, length=150):
    """
    From https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation

    This function computes the statistical autocorrelation of the data with itself with different time differences.

    Parameters:
    - x (numpy array): Array with the data
    - length (int, optional): maximum difference in time steps (Years) to be looked at.


    Returns:
    - result (numpy array): array of correlation coefficients

    """
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  \
        for i in range(1, length)])

def f(m,b,start,end):
    values = m*np.arange(start,end,1)+b
    return values


#-------------------------
# Plotting
#-------------------------


# Linear Regression Forecast and Lead time Analysis

start_year = 1850
end_year = 2025
window_length = 10
num_ly = 10
predict,starts = pf.linear_reg_pred(df,start_year,end_year,window_length = window_length)
correlation = []

gs_kw = dict(width_ratios=[2.5, 1], height_ratios=[1, 2])
fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                               ['lower left', 'right']],
                              gridspec_kw=gs_kw, figsize=(10, 4),
                              layout="constrained"
                              ,dpi = 300)

axd["upper left"].plot(df["Year"],df["Tropical Storms"], label = "Data")
axd["upper left"].set_xlabel("Year")
axd["upper left"].set_ylabel("Number of Events")
axd["upper left"].xaxis.set_visible(False)
axd["upper left"].legend()


for i in np.arange(5):
    ly_i = [prediction[i] for prediction in predict]
    correlation.append(np.corrcoef(ly_i,df["Tropical Storms"][0:len(ly_i)])[0,1])
    axd["lower left"].plot(np.arange(len(ly_i))+starts[i],ly_i, label = f"Lead Year {i}", alpha = 1-(i*2)/10)
axd["lower left"].set_xlabel("Year")
axd["lower left"].set_ylabel("Number of Events")
axd["lower left"].legend()


axd["right"].scatter(np.arange(len(correlation)),correlation)
axd["right"].set_xlabel("Lead Year")
axd["right"].set_ylabel("Pearson Correlation")
#axd["right"].legend()

fig.suptitle('Data and Lead Year Analysis for linear regression Forecast')

plt.savefig(f"final_plots/pdf/Lead_year_regression_forecast.pdf")
plt.savefig(f"final_plots/jpg/Lead_year_regression_forecast.jpg")