# -*- coding: utf-8 -*-
"""2023-07-14-vis-feature-importance.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1R9c4bCant6uRejnxjpj_h2T90X5khWvS
"""

import os
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Visualization and Plotting Setup
sns.set_theme(style="whitegrid")
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

tableau20 = [(r / 255., g / 255., b / 255.) for r, g, b in tableau20]

# Load Data
countFIS = pd.read_csv("/content/drive/MyDrive/G06-SolarDeployment/z-github-shared/featureimportance_count.csv")
# feature importance from 8 PV count models are aggregated using weights using the R^2 from each model

# Data restructure for bar graphs
countFIS["XGBoost Std"] = countFIS[['Random Forest', 'CATBoost', 'Light GBM', 'XGBoost']].sum(axis=1, skipna=True)
countFIS["Light GBM Std"] = countFIS["Random Forest"] + countFIS["Light GBM"]
countFIS["CATBoost Std"] = countFIS["Light GBM Std"] + countFIS["CATBoost"]
countFIS = countFIS[["Feature", "Random Forest", "Light GBM Std", "CATBoost Std", "XGBoost Std"]]
countFIS = countFIS.rename(columns={"XGBoost Std": "XGBoost", "Light GBM Std": "Light GBM", "CATBoost Std": "CATBoost"})

# Order by XGBoost feature importance
countFIS = countFIS.sort_values("XGBoost", ascending=False)

f, ax = plt.subplots(figsize=(9, 21))

# Plot the bar graphs
sns.barplot(x="XGBoost", y="Feature", data=countFIS, label="XGBoost", color=tableau20[12])
sns.barplot(x="CATBoost", y="Feature", data=countFIS, label="CATBoost", color=tableau20[4])
sns.barplot(x="Light GBM", y="Feature", data=countFIS, label="LightGBM", color=tableau20[18])
sns.barplot(x="Random Forest", y="Feature", data=countFIS, label="Random Forest", color=tableau20[2])

ax.legend(ncol=1, loc="lower right", frameon=True, fontsize=20)
ax.set_xlabel('Feature Importance', fontsize=14)
ax.set_ylabel('', fontsize=14)

# Change font size for x axis
ax.xaxis.get_label().set_fontsize(15)
ax.yaxis.get_label().set_fontsize(15)
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=13)

plt.show()