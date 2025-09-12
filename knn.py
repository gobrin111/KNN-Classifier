import pandas as pd
import numpy as np
from collections import Counter


class KNN:
    def __init__(self):
        self.l_test = None
        self.f_test = None
        self.l_train = None
        self.f_train = None
        self.labels = None
        self.k = None
        self.features = None
    
    def train(self, file, k):
        df = pd.read_excel(file, header=[0,1]) # had to define the header because there are two headers in this file
        self.k = k
        
        self.features = df.iloc[:, :-1].values   # first columns except last
        self.labels = df.iloc[:, -1].values    # last column = class
        
        np.random.seed(42)  # reproducibility, set to 42
        indices = np.arange(len(self.features))
        np.random.shuffle(indices)
        
        split = int(0.8 * len(self.features))
        train_i, test_i = indices[:split], indices[split:]

        self.f_train, self.l_train = self.features[train_i], self.labels[train_i]
        self.f_test, self.l_test   = self.features[test_i], self.labels[test_i]

    def test(self):
        predictions = []

        for test_point in self.f_test:
            # numpy vector distance calc, just makes the calculation faster when I'm running this
            distances = np.linalg.norm(self.f_train - test_point, axis=1)

            # indices for the nearest nodes based on k
            k_i = np.argsort(distances)[:self.k]

            # gets the labels based on the indexes what I saved in k_i
            k_labels = self.l_train[k_i]

            # gets the class based on majority
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)


knn = KNN()
# enter file name of xlsx and the k number that you want
knn.train("DataHW2.xlsx", 5)
predictions = knn.test()
print(predictions)

