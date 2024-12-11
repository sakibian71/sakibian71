import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Given data
ring_number = [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
D_square = [207.36, 193.21, 189.42, 178.3, 164.35, 159.76, 144.96, 138.56, 129.55, 114.92, 106.09, 92.65, 88.75, 77.34, 66.26]

# Convert to DataFrame
data = pd.DataFrame({'ring_number': ring_number, 'D_square': D_square})

# Split the data into training and testing sets
X = data[['ring_number']]
y = data['D_square']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Coefficients and intercept of the model
print(f"Coefficient: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Generate the line of best fit over the entire range
x_full = np.array(ring_number).reshape(-1, 1)  # Ensure it's a 2D column array
y_full = model.predict(x_full)

# Plotting the data and the linear regression line
plt.scatter(data['ring_number'], data['D_square'], color='black', label='Data Points')
plt.plot(x_full, y_full, color='red', linewidth=1, label='Linear Regression Line (Full Length)')

# Customize plot
plt.title('Linear Regression on $D^2$ vs. Ring Number', fontsize=14, fontweight='bold')  # Bold and big title
plt.xlabel('Ring Number', fontsize=12)
plt.ylabel('$D^2$', fontsize=12)
plt.legend()

# Save plot as PDF with specified DPI
plt.savefig('linear_regression_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()
