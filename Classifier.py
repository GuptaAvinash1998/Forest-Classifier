import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
from sklearn import preprocessing


os.chdir("D:\\Documents\\Pycharm_projects\\Projects\\Project2")
warnings.filterwarnings("ignore")  # ignores any given warning that gets generated
train_frame = pd.read_csv('train.csv')  # reads the training file
test_frame = pd.read_csv('test.csv')  # reads the testing file

labels_file = open("labels.txt", 'w')  # makes a file that stores the predicted labels

i = 0  # temp variable

size = len(train_frame.columns)
corealtions = []  # stores the calculated correlations

while i < size-1:  # calculated the correlation between the every single column and the labels column
    corealtions.append(train_frame[train_frame.columns[i]].corr(train_frame['2']))  # This is done so that it shows the columns that have a very low correlation
    i += 1  # we do this so that we drop the required column(s)


train_frame = train_frame.drop(columns=['0.29'])
# drops columns with low correlation

i = 0  # does the same thing for the testing data
size = len(test_frame.columns)
corealtions = []

while i < size:
    corealtions.append(test_frame[test_frame.columns[i]].corr(test_frame['3295']))
    i += 1


test_frame = test_frame.drop(columns=['0.28'])

X = train_frame.values[:, 0:52]  # separates the data and the labels
Y = train_frame.values[:, 53]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)  # spilts the data on a 70:30 split

sc = StandardScaler(with_mean=True, with_std=True)  # Since the data consists of mostly 0's and 1's, we standardize the data so that we get all the columns to the same range

X_train = sc.fit_transform(X_train)  # we standardize the training data
X_train = pd.DataFrame(X_train)

X_test = sc.transform(X_test)   # we standardize the testing data
X_test = pd.DataFrame(X_test)

'''X_train = (X_train - X_train.mean())/X_train.std()
X_test = (X_test - X_test.mean())/X_test.std()'''

d_tree = DecisionTreeClassifier(criterion="entropy", max_depth=1100, min_samples_leaf=16, random_state=100)  # the classifier is a decision tree using entropy and information gain
d_tree.fit(X_train, Y_train)  # fits the training and testing data

predictions = d_tree.predict(X_test)  # makes the predictions for the testing data
print(f1_score(Y_test, predictions, average='micro') * 100)  # calculates the accuracy using f1 score

test_frame = sc.fit_transform(test_frame)  # does the same thing for the testing file, it standardizes the testing file for easier prediction
test_frame = pd.DataFrame(test_frame)

# test_frame = (test_frame - test_frame.mean())/test_frame.std()
x = test_frame.values[:, 0:52]
predictions = d_tree.predict(x)  # makes a prediction for all the rows in the testing file and stores it in a list

i = 0
while i < predictions.size:  # loops through the list and writes to the file for submission
    labels_file.write(str(predictions[i]) + "\n")
    i += 1

labels_file.close()  # after finishing the writing, we close the file