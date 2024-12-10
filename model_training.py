import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import glob 
import os
from sklearn import metrics
from sklearn.ensemble import AdaBoostRegressor
import numpy as np



scenarios = 'Project/data/scenarios/new_sc.csv'
# Sort scenarios by name
# Load the data and train the model for each scenario
df = pd.read_csv(f'{scenarios}')
df.drop(columns='Unnamed: 0', inplace=True)

# Ignore outliers for actual
Q1 = df['GPA'].quantile(0.25)
Q3 = df['GPA'].quantile(0.75)
IQR = Q3 - Q1

# Define the bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
df = df[(df['GPA'] >= -(lower_bound)) & (df['GPA'] <= upper_bound)]

# Split the data
y = df['GPA']
X = df.drop(columns=['GPA'])
    
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
 
# Train the random forest
clf = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=42)
clf.fit(X_train, y_train)
# Predict the data
y_pred_rf = clf.predict(X_test)
    

# Group actual and predicted values by percentiles
data = pd.DataFrame({'actual': y_test, 'predicted': y_pred_rf})
# Create a column 'percentile' which represents the percentile of the predicted price in the test data set
data['percentile'] = pd.qcut(y_pred_rf, [0, 0.25, 0.50, 0.75, 0.90, 0.95, 1], labels=False)
    
    #data['percentile'] = np.percentile(y_pred_rf, [25, 50, 75, 90, 95])
    # # Train AdaBoost
    # ada_model = AdaBoostRegressor(n_estimators=100, random_state=42)
    # ada_model.fit(X, y)   
    
    # # Predict the data
    # y_pred_ada = ada_model.predict(X_test)

    # Write the metrics
percentile_info = []
with open(f'new_sc.txt', 'w+') as output:
    output.write(f'RandomForest: \n Scenario: new_sc, \n RMSE: {metrics.root_mean_squared_error(y_test, y_pred_rf)} \n R squared: {metrics.r2_score(y_test, y_pred_rf)} \n MAPE: {metrics.mean_absolute_percentage_error(y_test, y_pred_rf)} \n\n')
    # output.write(f'AdaBoost: \n Scenario: {scenario}, \n RMSE: {metrics.root_mean_squared_error(y_test, y_pred_ada)} \n R squared: {metrics.r2_score(y_test, y_pred_ada)} \n MAE: {metrics.mean_absolute_error(y_test, y_pred_ada)} \n')
    # Calculate metrics per percentile
    for p in sorted(data['percentile'].unique()):
        # Plot the mse, mape and r2 of each percentile
        mse = metrics.mean_squared_error(data.loc[data['percentile'] == p, 'actual'], data.loc[data['percentile'] == p, 'predicted'])
        actual = data.loc[data['percentile'] == p, 'actual']
        mape = metrics.mean_absolute_percentage_error(data.loc[data['percentile'] == p, 'actual'], data.loc[data['percentile'] == p, 'predicted'])
        r2 = metrics.r2_score(data.loc[data['percentile'] == p, 'actual'], data.loc[data['percentile'] == p, 'predicted'])
        percentile_info = percentile_info + [{'percentile': p, 'MSE' : mse, 'MAPE': mape, 'R2': r2}]
        output.write(f'Percentile {p}: \n RMSE: {np.sqrt(mse)}, \n MAE: {mape}, \n R2: {r2} \n\n')
        
        
        
# Plot the percentile_info
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
print([ p['percentile'] for p in percentile_info])
# Plot each metric
plt.plot([ p['percentile'] for p in percentile_info], [ p['MSE'] for p in percentile_info ], label='MSE', marker='o')
plt.plot([ p['percentile'] for p in percentile_info], [ p['MAPE'] for p in percentile_info ], label='MAPE', marker='o')
plt.plot([ p['percentile'] for p in percentile_info], [ p['R2'] for p in percentile_info ], label='R²', marker='o')

# Adding labels and title
plt.xlabel('Percentile')
plt.ylabel('Metric Value')
plt.title('Metrics Across Percentiles')
plt.legend()

# Show grid and plot
plt.grid(True)
plt.show()
        

