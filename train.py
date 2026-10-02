import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns


df = pd.read_csv("nvidia.csv")
df.head()

df.shape()

df.size()

df.ndim()

df.corr()

df.isnull().sum()

df["date"] = pd.to_datetime(df["date"], format="%d%m%Y")

df.head()


data = df.sort_values("date").reset_index(drop=True)


data.head()

plt.figure(figsize=(15, 6))

plt.plot(data["date"], data["close"])
plt.title("Closing price over time")
plt.xlabel("Date ")
plt.ylabel("Closing Price")
plt.xticks(rotations=45)
plt.tight_layout()
plt.show



print(data["close"]).describe()


columns = ["date",	"open",	"high",	"low",	"close",	"volume","shares_outstanding_bn",	"market_cap_usd_bn",	"quarterly_revenue_usd_bn",	"sma_20"]
for columns in columns:
    plt.figure(figsize=(15,5))
    plt.plot(data["date"], data[columns])
    plt.title(f"{columns} Over time")
    plt.xlabel("Date")
    plt.ylabel(columns)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



data.isnull().sum()

correlation = data.select_dtypes("number").corr()["close"].sort_values(ascending=False)


# after visualizing the correaltion heatmap now we need to remove the unnecessary features
removed_column = [
    'open', 
    'high', 
    'low', 
    'sma_20', 
    'market_cap_usd_bn', 
    'shares_outstanding_bn', 
    'quarterly_revenue_usd_bn'
]

data_clean = data.drop(columns=removed_column, errors = 'ignore')
print("Remaining columns:", data_clean.columns.tolist())
data_clean.head()



print(data_clean.head())
print(data_clean.describe())


plt.figure(figsize=(15, 5))
plt.plot(data_clean["date"], data_clean["volume"])
plt.tight_layout()
plt.title("Trading volume over time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.xticks(rotations=45)
plt.show()


Q1 = data_clean["close"].quantile(0.25)
Q3 = data_clean["close"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


outliers = data_clean[
    (data_clean["close"] < lower_bound) |
    (data_clean["close"] > upper_bound)

]

print("Number of outliers:", len(outliers))
print(outliers.head())

print(data_clean.columns)
print(data_clean["date"].is_monotonic_increasing)
print("Duplicate data:", data_clean["date"].duplicated().sum())



n = len(data_clean)
print(n)


train_end = int(n * 0.80)
validation_end = int(n * 0.90)

train = data_clean.iloc[:train_end].copy()
validation = data_clean.iloc[train_end:validation_end].copy()
test = data_clean.iloc[validation_end:].copy()

print("Total:", len(data_clean))
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


print("\nTrain:")
print(train["date"].min(), "-", train["date"].max())

print("\nValidation:")
print(validation["date"].min(), "-", validation["date"].max())

print("\nTest:")
print(test["date"].min(), "-", test["date"].max())



#Training the model now

prophet_train = train[["date", "close"]].copy()

prophet_train = prophet_train.rename(columns={
    "date": "ds",
    "close": "y"
})

print(prophet_train.head())
print(prophet_train.shape)

model = Prophet()

model.fit(prophet_train)

future = model.make_future_dataframe(periods=30)

# 2. Predict trends and seasonalities
forecast = model.predict(future)

# 3. Plot the results
fig = model.plot(forecast)

prophet_validation = validation[["date", "close"]].copy()

prophet_validation = prophet_validation.rename(columns={
    "date": "ds",
    "close": "y"
})

forecast = model.predict(
    prophet_validation[["ds"]]
)

print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].head())

comparison = validation[["date", "close"]].copy()

comparison["predicted"] = forecast["yhat"].values

comparison = comparison.rename(columns={
    "date": "ds",
    "close": "actual"
})

print(comparison.head())

mae = mean_absolute_error(
    comparison["actual"],
    comparison["predicted"]
)

rmse = np.sqrt(mean_squared_error(
    comparison["actual"],
    comparison["predicted"]
))

mape = np.mean(
    np.abs(
        (comparison["actual"] - comparison["predicted"])
        / comparison["actual"]
    )
) * 100

print("Prophet Validation Results")
print("---------------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("MAPE:", mape, "%")

plt.figure(figsize=(14, 6))

plt.plot(
    comparison["ds"],
    comparison["actual"],
    label="Actual"
)

plt.plot(
    comparison["ds"],
    comparison["predicted"],
    label="Prophet Prediction"
)

plt.title("Prophet: Actual vs Predicted - Validation Set")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)


print("Validation actual minimum:", comparison["actual"].min())
print("Validation actual maximum:", comparison["actual"].max())
print("Validation actual mean:", comparison["actual"].mean())


model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)

model.fit(prophet_train)

forecast = model.predict(
    prophet_validation[["ds"]]
)