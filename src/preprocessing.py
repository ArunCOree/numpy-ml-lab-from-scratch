import numpy as np
from utils import shuffle_data, add_bias_term

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
                    new_row.append(np.nan)   
                else:
                    new_row.append(float(value))  
            cleaned_x.append(new_row)

        self.x = np.array(cleaned_x, dtype=float)  

    def fill_missing_values(self):
        for i in range(self.x.shape[1]):
            column = self.x[:, i]
            mean_value = np.nanmean(column)  

            
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

        
        std = np.where(std == 0, 1, std)

        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

        return train_x, test_x

    def process(self):
        self.handling_missing_values()
        self.fill_missing_values()
        self.x,self.y = shuffle_data(self.x, self.y)
        train_x, train_y, test_x, test_y = self.train_test_split()
        train_x, test_x = self.normalize(train_x, test_x)

        return train_x, train_y, test_x, test_y


