#importing required libraries
import pandas as pd
import numpy as np
from pandasgui import show

#importing Climate Data into temporary memory
weather= pd.read_csv("Climate_Data.csv", index_col="DATE")

# Convert the index to a datetime object and then format it to the desired format
original_format = "%Y-%m-%d"
desired_format = "%m-%d-%Y"
weather.index = pd.to_datetime(weather.index, format=original_format).strftime(desired_format)


#Data Cleaning and Formatting
weather.apply(pd.isnull).sum()/weather.shape[0]
core_weather =weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]
core_weather.apply(pd.isnull).sum()/core_weather.shape[0]
core_weather["snow"] = core_weather["snow"].fillna(0)
core_weather["snow_depth"] = core_weather["snow_depth"].fillna(0)
core_weather["precip"]=core_weather["precip"].fillna(0)
core_weather[pd.isnull(core_weather["temp_min"])]
core_weather=core_weather.fillna(method="ffill")
core_weather.apply(pd.isnull).sum()/weather.shape[0]
core_weather.index=pd.to_datetime(core_weather.index)
print(core_weather.dtypes)
print(core_weather.index)
core_weather["target"]=core_weather.shift(-1)["temp_max"]
core_weather=core_weather.iloc[:-1,:].copy()
# core_weather= core_weather.to_csv("Climate_Data.csv", index="FALSE")


#training our first ML model:
from sklearn.linear_model import Ridge
reg=Ridge(alpha=.1)
predictors=["precip", "temp_max", "temp_min", "snow", "snow_depth"]
train=core_weather.loc[:"2020-12-31"]
test=core_weather.loc["2021-01-01":]
reg.fit(train[predictors], train["target"])
Ridge(alpha=0.1)
predictions=reg.predict(test[predictors])
from sklearn.metrics import mean_absolute_error
error=mean_absolute_error(test["target"],predictions)
print(error)
combined=pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns=["actual","predictions"]
print(combined)

#defining function
def create_predictions(predictors, core_weather, reg):
    train = core_weather.loc[:"2020-12-31"]
    test = core_weather.loc["2021-01-01":]
    reg.fit(train[predictors], train["target"])
    predictions = reg.predict(test[predictors])
    error=mean_absolute_error(test["target"], predictions)
    combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
    combined.columns = ["actual", "predictions"]
    return error, combined

core_weather["month_max"]=core_weather["temp_max"].rolling(30).mean().astype(np.float64)
core_weather["month_day_max"]=core_weather["month_max"]/(core_weather["temp_max"]+ 1e-8).astype(np.float64)
core_weather["max_min"]=core_weather["temp_max"]/(core_weather["temp_min"]+ 1e-8).astype(np.float64)
predictors=["precip", "temp_max", "temp_min", "snow", "snow_depth", "month_max", "month_day_max", "max_min"]
core_weather=core_weather.iloc[30:,:].copy()
# print(core_weather["month_max"].describe())
# print(core_weather["month_day_max"].describe())
# print(core_weather["max_min"].describe())

error, combined=create_predictions(predictors, core_weather, reg)
print(error)
print(combined)
# print(combined.plot())
# plt.show()
core_weather["monthly_avg"] = core_weather.groupby(core_weather.index.month)["temp_max"].transform(lambda x: x.expanding().mean())
core_weather["day_of_year_avg"]=core_weather.groupby(core_weather.index.day_of_year)["temp_max"].transform(lambda x: x.expanding().mean())
predictors=["precip", "temp_max", "temp_min", "snow", "snow_depth", "month_max", "month_day_max", "max_min", "monthly_avg", "day_of_year_avg"]
error, combined=create_predictions(predictors, core_weather, reg)
print(error)
print(core_weather)
print(reg.coef_)
print(core_weather.corr()["target"])
combined["diff"]=(combined["actual"]-combined["predictions"]).abs()
print(combined.sort_values("diff", ascending=False).head())
print(core_weather)

#Showing actual and predictions climate data
show(core_weather)
show(combined)
