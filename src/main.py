import numpy as np
from preprocessing import Preprocessor
from linear_model import LinearRegression
from metrics import mean_squared_error, mean_absolute_error, r2_score

def load_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=str)
    x = data[:, :-1]  
    y = data[:, -1].astype(float)  
    return x, y

def main():
    x, y = load_csv('data.csv')
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

main()

if __name__ == "__main__":
    main()