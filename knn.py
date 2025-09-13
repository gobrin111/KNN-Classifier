import pandas as pd
import numpy as np
from collections import Counter

# the runner lines are after the class
# just change the stuff at the bottom to test other data and k numbers

# this code scales for the number of features and classes, but I'm assuming that the classes are still from 0 to whatever number iteratively
# and there are still two headers

class KNN:
    def __init__(self):
        self.l_test = None # classes for the test data
        self.f_test = None # features for the test data
        self.l_train = None
        self.f_train = None
        self.labels = None
        self.k = None
        self.features = None
        self.classes = None
        self.predictions = None
    
    def train(self, file, k):
        df = pd.read_excel(file, header=[0,1]) # had to define the header because there are two headers in this file
        self.k = k  # set the k number from user input
        self.classes = df.columns[-1][1] # gets class names in the last column of second row, assuming that other class names are the same structure as the example given

        # processing the class string into a dictionary, so I can call it later for the print outs
        self.classes = self.classes.split(', ')
        newClasses = {}
        for cls in self.classes:
            numWord = cls.split(":")
            newClasses[int(numWord[0])] = numWord[1]
        self.classes = newClasses
        # splits the features from their labels, but order is still maintained
        self.features = df.iloc[:, :-1].values   # first columns except last = features
        self.labels = df.iloc[:, -1].values    # last column = class
        
        np.random.seed()  # reproducibility, set to 42
        indices = np.arange(len(self.features))
        np.random.shuffle(indices)
        
        split = int(0.8 * len(self.features))
        train_i, test_i = indices[:split], indices[split:]

        self.f_train, self.l_train = self.features[train_i], self.labels[train_i]
        self.f_test, self.l_test   = self.features[test_i], self.labels[test_i]

    def test(self):
        self.predictions = []

        for test_point in self.f_test:
            # numpy vector distance calc, just makes the calculation faster when I'm running this
            distances = np.linalg.norm(self.f_train - test_point, axis=1)

            # sort indices for the nearest nodes based on k
            k_i = np.argsort(distances)[:self.k]

            # gets the labels based on the indexes what I saved in k_i
            k_labels = self.l_train[k_i]

            # gets the class based on majority
            most_common = Counter(k_labels).most_common(1)[0][0]
            self.predictions.append(most_common)

        return np.array(self.predictions)

    def evaluate(self):
        # gets the unique classes in number form sort from 0 to .....
        actual_classes = sorted(self.classes.keys())
        n_classes = len(actual_classes)

        # build confusion matrix
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(self.l_test, self.predictions):
            confusion[true][pred] += 1 # true pred form the coordinate for where to increment

        # calculate accuracy
        # trace gets me the diagonal of all the trues
        # sum gets me the sum of the whole matrix
        accuracy = np.trace(confusion) / np.sum(confusion)

        # print out k number and the matrix header
        print(f"\nK = {self.k}, Confusion matrix is:")
        # print confusion matrix
        for row in confusion:
            print(" ".join(str(x) for x in row))
        # print text below matrix
        print(f"Accuracy = {accuracy}")

        # sensitivity and precision calc and print out
        for cls in actual_classes:
            TP = confusion[cls, cls]
            FN = np.sum(confusion[cls, :]) - TP
            FP = np.sum(confusion[:, cls]) - TP
            TN = np.sum(confusion) - (TP + FN + FP)
            # if zero is both then just output zero since 0/0 causes issue
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0

            print(f"Sensitivity {self.classes[int(cls)]} = {sensitivity}")
            print(f"Precision {self.classes[int(cls)]} = {precision}")


# the code is run through here, make changes here to test other stuff
knn = KNN()
# enter file name of xlsx and the k number that you want
knn.train("DataHW2.xlsx", 10)
predictions = knn.test()
print(predictions)
knn.evaluate()

knn.train("DataHW2.xlsx", 15)
predictions = knn.test()
print(predictions)
knn.evaluate()

knn.train("DataHW2.xlsx", 50)
predictions = knn.test()
print(predictions)
knn.evaluate()


