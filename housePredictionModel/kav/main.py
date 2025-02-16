import numpy as np
import pandas as pd
import re
df = pd.read_csv("C:\SID_DATA\SIDDHARTH\UPES COLLEGE STUDY MATERIAL\SEM6\AIML\LAB\housePredictionModel\kav/Bengaluru_House_Data.csv")
df
df['total_sqft'] 
df.isnull().sum()
print(df['area_type'].value_counts())

df.dropna(subset=['location'])
df['size'] = df['size'].str.extract('(\d+)').astype(float)  # Extract numbers from "2 BHK" format
df['size'].fillna(df['size'].median(), inplace=True)
print(df['size'].value_counts())
df.drop(columns=['society'], inplace=True)
df['bath'].fillna(df['bath'].median(), inplace=True)
df
df['balcony'].fillna(0, inplace=True)
df
print(df.nunique())
df['availability'] = df['availability'].apply(lambda x: 1 if x == 'Ready To Move' else 0)
df = pd.get_dummies(df, columns=['area_type'], drop_first=True)
location_counts = df['location'].value_counts().to_dict()
df['location'] = df['location'].map(location_counts)
df['area_type_Carpet  Area'] = df['area_type_Carpet  Area'].apply(lambda x: 1 if x == True else 0)
df['area_type_Plot  Area'] = df['area_type_Plot  Area'].apply(lambda x: 1 if x == True else 0)
df['area_type_Super built-up  Area'] = df['area_type_Super built-up  Area'].apply(lambda x: 1 if x == True else 0)
df = df[pd.to_numeric(df['total_sqft'], errors='coerce').notna()]
df
df = df.dropna()
X = df.drop(columns=['price'])  # Features
y = df['price']  # Target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Assuming X_train, y_train, X_test, y_test are already defined
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
# Predict on test data
y_pred = model.predict(X_test)
y_pred
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Predict on test data
y_pred = model.predict(X_test)

# Visualize the output
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_test, color='blue', label='Actual Prices')
sns.scatterplot(x=y_test, y=y_pred, color='red', label='Predicted Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()