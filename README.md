# Weather_Prediction_project
This is my first ML related project. The dataset used is given (Climate_data.csv) and the code is in File1.py <br>

Project Description: <br>

This Python project utilizes machine learning to predict climate data based on historical weather information. The code provided performs the following key tasks:
 <br>
Data Import and Formatting:
 <br>
The project begins by importing climate data from a CSV file into a Pandas DataFrame. The date index is converted to a desired format.
Data cleaning and formatting are performed to handle missing values in precipitation, snowfall, snow depth, and temperature. <br>
Machine Learning Model: <br>

A Ridge Regression model is trained using the scikit-learn library. It aims to predict the maximum temperature (temp_max) for a given day based on various predictors, including precipitation, maximum temperature (temp_max), minimum temperature (temp_min), snowfall (snow), snow depth (snow_depth), and derived features like monthly averages and day-of-year averages. <br>
Evaluation: <br>

The model's performance is evaluated using the Mean Absolute Error (MAE) metric to assess the accuracy of temperature predictions. <br>
Feature Engineering:
 <br>
Additional features are engineered, such as rolling monthly maximum temperature averages (month_max), the ratio of monthly maximum temperature to daily maximum temperature (month_day_max), and the ratio of maximum temperature to minimum temperature (max_min). <br>
Prediction Function:
 <br>
A reusable function, create_predictions, is defined to streamline the training and prediction process with different sets of predictors. <br>
Data Visualization: <br>

The project also includes commented-out code for data visualization using Matplotlib (plot and show). You can uncomment and use this code to visualize the actual and predicted climate data. <br>
Data Exploration: <br>

Correlation analysis and data exploration are performed to understand the relationships between predictors and the target variable (temp_max). <br>
Display of Results: <br>

The project displays the results, including temperature predictions and the largest prediction errors, to help analyze the model's performance. <br>
This project provides a foundation for predicting climate data using machine learning techniques and can be further extended to explore more advanced algorithms and features. <br>
