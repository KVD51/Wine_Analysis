Wine_Analysis is a program that analyzes the wine dataset provided by sklearn. The wine dataset contains 178 wine samplings, each with measurements on 13 different attributes/variables. Each sample is categorized into three classes: class 0, 1, 2.

The purpose of the analysis is to illustrate knowledge and use of data science libraries. Libraries used are: pandas, numpy, matplotlib, seaborn, and scikit-learn (sklearn).

Analysis performed:
I started without knowing anything about the variables contained in the dataset - not knowing variable definitions or what they might indicate. I wanted to first import the dataset, then look for any potential relationships between variables, then try using some plotting and statistical techniques to illustrate anything observed in the data. After discovering some patterns, I looked up the definitions of certain variables to give me insight into the patterns.

techniques:

-Imported the dataset into a panda dataframe, then viewed the frame head to get a sense of the data for each variable.

-Grouped the dataframe by class to view the average value by class for each variable.

-Created seaborn barplots of specific variables that appeared to have a relationship - by data and name.

-Created seaborn kde plots to view the distributions of a variable that appeared to indicate a strong difference between classes.

-Created a seaborn scatterplot of two variables to get a further sense of the difference between the classes.

-Used linear regression functions from sklearn to add a regression line to the scatterplot - did this to illustrate the apparent relationship between variables.

The program can be run with wine_analysis.py or use wine_analysis.ipynb to view a jupyter notebook that includes code and brief commentary.
