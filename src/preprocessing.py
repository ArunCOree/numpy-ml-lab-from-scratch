import numpy as np

class Preprocessor:
    def __init__(self, x, y):
        self.x = x
        self.y = y  

    def handling_missing_values(self):
        cleaned_x = []
        for row in self.x:
            new_row = []
            for value in row:
                if value is None or value == "":
                    new_row.append(np.nan)   # Use NaN instead of 0
                else:
                    new_row.append(float(value))  # Convert everything to float
            cleaned_x.append(new_row)

        self.x = np.array(cleaned_x, dtype=float)  # Force numeric type

    def fill_missing_values(self):
        for i in range(self.x.shape[1]):
            column = self.x[:, i]
            mean_value = np.nanmean(column)  # Ignore NaN

            # Replace NaN with mean
            self.x[:, i] = np.where(np.isnan(column), mean_value, column)

    def train_test_split(self, test_size=0.2):
        total_samples = self.x.shape[0]
        test_samples = int(total_samples * test_size)

        indices = np.arange(total_samples)
        np.random.shuffle(indices)

        self.x = self.x[indices]
        self.y = self.y[indices]

        train_x = self.x[:-test_samples]
        train_y = self.y[:-test_samples]
        test_x = self.x[-test_samples:]
        test_y = self.y[-test_samples:]

        return train_x, train_y, test_x, test_y

    def normalize(self, train_x, test_x):
        mean = np.mean(train_x, axis=0)
        std = np.std(train_x, axis=0)

        # Prevent division by zero
        std = np.where(std == 0, 1, std)

        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

        return train_x, test_x

    def process(self):
        self.handling_missing_values()
        self.fill_missing_values()
        train_x, train_y, test_x, test_y = self.train_test_split()
        train_x, test_x = self.normalize(train_x, test_x)

        return train_x, train_y, test_x, test_y
# Features (X)
x = [
    [25, 50000, 2],
    [30, "", 3],
    [None, 60000, 2],
    [35, 80000, None],
    [40, 90000, 4],
    ["", 100000, 3],
    [28, None, 2],
    [32, 70000, ""],
    [45, 120000, 5],
    [None, None, None],

    [29, 52000, 2],
    [31, "", 3],
    [27, 61000, 2],
    [36, 82000, None],
    [42, 91000, 4],
    ["", 110000, 3],
    [26, None, 2],
    [33, 71000, ""],
    [46, 125000, 5],
    [None, None, None],

    [24, 48000, 1],
    [34, "", 3],
    [None, 65000, 2],
    [37, 83000, None],
    [41, 92000, 4],
    ["", 105000, 3],
    [29, None, 2],
    [30, 72000, ""],
    [44, 118000, 5],
    [None, None, None]
]

# Labels (y)
y = [
    0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
    0, 1, 0, 1, 1, 0, 0, 1, 1, 0
]
x = np.array(x, dtype=object)
y = np.array(y)
preprocessor = Preprocessor(x, y)
train_x, train_y, test_x, test_y = preprocessor.process()
print("Train X:\n", train_x)
print("Train Y:\n", train_y)
print("Test X:\n", test_x)
print("Test Y:\n", test_y)