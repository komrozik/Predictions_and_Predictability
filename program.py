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

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#-------------------------
# Plotting
#-------------------------


# Linear Regression Forecast and Lead time Analysis

start_year = 1850
end_year = 2025
window_length = 5
num_ly = 10
predict,starts = pf.linear_reg_pred(df,start_year,end_year,window_length = window_length)
correlation = []
rmses = []

gs_kw = dict(width_ratios=[2.5, 1])
fig, axd = plt.subplot_mosaic([['left', 'right']],
                              gridspec_kw=gs_kw, figsize=(10, 4),
                              layout="constrained"
                              ,dpi = 300)


# axd["left"].set_xlabel("Year")
# axd["left"].set_ylabel("Number of Events")
# axd["left"].xaxis.set_visible(False)
# axd["left"].legend()


for i in np.arange(window_length,window_length+5,1):
    ly_i = [prediction[i] for prediction in predict]
    correlation.append(np.corrcoef(ly_i[0:-i],df["Tropical Storms"][i:len(ly_i)])[0,1])
    rmses.append(rmse(ly_i[0:-i],df["Tropical Storms"][i:len(ly_i)]))
    axd["left"].plot(np.arange(len(ly_i))+starts[i],ly_i, label = f"Lead Year {i-window_length}", alpha = 1-((i-window_length)*2)/10)
axd["left"].set_xlabel("Year")
axd["left"].set_ylabel("Number of Events")

axd["left"].plot(df["Year"],df["Tropical Storms"], label = "Data", color = "red")

axd["left"].legend()



color = 'tab:blue'
axd["right"].plot(np.arange(len(correlation)),correlation,color=color,marker = "o")
axd["right"].set_xlabel("Lead Year")
axd["right"].set_ylabel("Pearson Correlation",color = color)
axd["right"].tick_params(axis='y', labelcolor=color)

ax2 = axd["right"].twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:orange'
ax2.set_ylabel('RMSE',color = color)  # we already handled the x-label with ax1
ax2.plot(np.arange(len(rmses)),rmses,color = color,marker = "o")
ax2.tick_params(axis='y', labelcolor=color)
#axd["right"].legend()

fig.suptitle(f'Data and Lead Year Analysis for linear regression Forecast \n Training with {window_length} Years')


plt.savefig(f"plots/Lead_year_regression_forecast_{window_length}.pdf")
plt.savefig(f"plots/Lead_year_regression_forecast_{window_length}.jpg")

plt.savefig(f"final_plots/pdf/Lead_year_regression_forecast_{window_length}.pdf")
plt.savefig(f"final_plots/jpg/Lead_year_regression_forecast_{window_length}.jpg")