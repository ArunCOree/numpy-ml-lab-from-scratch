import numpy as np
from preprocessing import Preprocessor
from linear_model import LinearRegression
from metrics import mean_squared_error, mean_absolute_error, r2_score

x = np.array([[200,5.5,1200],
              [150,4.0,900],
              [300,6.0,1500],
              [250,5.0,1100],
              [180,4.5,1000]])

y = np.array([75,70,80,65,72,])

preprocessor = Preprocessor(x, y)
train_x, train_y, test_x, test_y = preprocessor.process()

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(train_x, train_y)

predictions = model.predict(test_x)
mse = mean_squared_error(test_y, predictions)
mae = mean_absolute_error(test_y, predictions)
r2 = r2_score(test_y, predictions)

print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
print(f'R^2 Score: {r2:.4f}')

new_car = np.array([[220,5.2,1300]])
pred = model.predict(new_car)
print(f'Predicted MPG for new car: {pred[0]:.2f}')