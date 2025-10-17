"""
Firstly, I loaded a sample of the dataset (100k rows) because the full dataset is very large.
Checked the description and found no outliers or need for feature scaling.

Then plotted w vs x vs y and observed that y depends on x in a sinusoidal pattern.

Finally, after training the Random Forest model, the actual vs predicted plot is almost a perfect diagonal line.
Errors were measured using RMSE and R² score.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# I did this part using chatgpt cuz i dont know how to handle files

train_path = r"C:\Users\LOCALACCOUNT\Downloads\archive\train_data.csv"
test_path  = r"C:\Users\LOCALACCOUNT\Downloads\archive\test_data.csv"


# Using a small part 100k rows of the data set cuz it is taking a lot of time to load the entire dataset
train_sample = pd.read_csv(train_path, nrows=100_000)
print(train_sample.head())
print(train_sample.info())

# Checking if there are outliers and ranges of each feature for further feature scaling if needed
print("\nDescriptive stats:")
print(train_sample.describe())

# Visuvalizing this data set
sns.pairplot(train_sample.sample(2000))   
plt.suptitle("w, x, y Relationships", y=1.02)
plt.show()
# By seeing this plots btw w vs x and w vs y the w has almost the same value for every data point which means w does not vary, hence it will contribute nothing to predicting y
# But the plot for x vs y is showing us y inc x dec its like a sinusodial wave the function might be some function of sin or cos 


# As there is nothing to remove or scale as there are no outliers or missing values or categorical values lets directly move to training part
X = train_sample[['w', 'x']]
y = train_sample['y']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# seeing the covariance btw w , x and y
print(train_sample[['w','x','y']].corr())
# Correlation shows y depends almost entirely on x, w contributes very little


# Rough approximation to see if it function is pie*sinx
from numpy import sin, pi
train_sample['y_approx'] = sin(train_sample['x'] * pi)
plt.scatter(train_sample['x'], train_sample['y'], alpha=0.3, label='actual y')
plt.scatter(train_sample['x'], train_sample['y_approx'], alpha=0.3, label='sin(x) approx')
plt.legend()
plt.show()
# as the plot shows it is almost the same LOL!
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(train_sample['y'], train_sample['y_approx'])
print(f"MAE of sinusoidal approximation: {mae:.4f}")
#The approximation is around 80% which means the function is almost the same 

# Im using RandomForest model because as its clearly not a linear plot its something like sinusodial we will use non-linear model to predict
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_val)

# Now lets visuvalize the plot for actual vs predicted 
plt.figure(figsize=(6, 6))
plt.scatter(y_val, y_pred, alpha=0.3)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')  # diagonal line
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Predicted vs Actual (Random Forest)")
plt.show()

# As the plot clearly shows that the model is very much accurate in giving the results its a diagonal stright line 



# using rmse for checking how much off are we from the actual values and r2 score to see how much my model model captures the trend
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f"Validation RMSE: {rmse:.4f}")
print(f"Validation R²:   {r2:.4f}")

# For this part of the dataset the error is very small around 0.06 and r2 score is 0.99 which means 99% of the targets variance is explained

# Conclusion:
# - y is mostly determined by x in a sinusoidal pattern.
# - Random Forest can accurately capture this non-linear relationship.
# - RMSE = 0.06 and R² = 0.99 confirm the model predicts very accurately.
