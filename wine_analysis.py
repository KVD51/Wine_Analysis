"""
Wine_Analysis is a program that analyzes the wine dataset provided by sklearn. The wine dataset contains 178 wine
samplings, each with measurements on 13 different attributes/variables. Each sample is categorized into three
classes: class 0, 1, 2.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_wine


# Load wine dataset from sklearn and load into to a panda dataframe
wine = load_wine()
df = pd.DataFrame(wine.data, columns= wine.feature_names)
targets = wine.target #these are the classes of each sample
df['wclass'] = targets

# Summarize data by target/class (there are three classes of wine: 0, 1, 2)
# Data represents the mean of each column  - 13 columns plus index for class
summary = df.groupby('wclass').mean().reset_index()
print(summary)

# Use Seaborn to make barplots of means for specific columns/attributes
sns.set_style('darkgrid')
sns.set_palette('Paired')
plt.subplot(1, 3, 1)
sns.barplot(data=df, x='wclass', y='total_phenols')
plt.subplot(1, 3, 2)
sns.barplot(data=df, x='wclass', y='flavanoids')
plt.subplot(1, 3, 3)
sns.barplot(data=df, x='wclass', y='nonflavanoid_phenols')
plt.subplots_adjust(wspace=0.8)
plt.suptitle('Phenols and Flavanoids By Class of Wine\n (error bar represents 95% confidence interval)')
plt.show()

# Use Seaborn to make KDE plots for the color_intensity of each class of wine
class0_color_intensity = df.apply(lambda row: row.color_intensity if row.wclass == 0 else None, axis=1)
class1_color_intensity = df.apply(lambda row: row.color_intensity if row.wclass == 1 else None, axis=1)
class2_color_intensity = df.apply(lambda row: row.color_intensity if row.wclass == 2 else None, axis=1)

sns.set_style('darkgrid')
sns.set_palette('pastel')
sns.kdeplot(data=class0_color_intensity, shade=True)
sns.kdeplot(data=class1_color_intensity, shade=True)
sns.kdeplot(data=class2_color_intensity, shade=True)
plt.title('Color Intensity Distribution By Class of Wine')
plt.legend(['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Color Intensity')
plt.ylabel('Density Function')
plt.show()

# Use Seaborn to make scatterplot of color_intensity vs hue, and add a regression line to the plot using sklearn
sns.scatterplot(x='color_intensity', y='hue', hue='wclass', style='wclass', palette='Set2', data=df)

color_intensity = df.color_intensity.values
color_intensity = color_intensity.reshape(-1,1)
w_hue = df.hue.values
w_hue = w_hue.reshape(-1,1)

fitter = LinearRegression()
fitter.fit(color_intensity, w_hue)
hue_predict = fitter.predict(color_intensity)

plt.plot(color_intensity, hue_predict, color='blue')
plt.title('Color Intensity And Hue Scatterplot, By Class of Wine')
plt.show()
