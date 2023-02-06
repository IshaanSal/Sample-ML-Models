import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";") #identifies the file being scanned as well as the character separating the data
data = data[["G1", "G2", "failures", "studytime", "absences", "G3"]] #the sample data being identified
predict = "G3"

x = np.array(data.drop([predict], 1)) #attributes (uses predict because this allows you to change the attribute being looked at)
y = np.array(data[predict]) #labels

best_acc = 0
for counter in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best_acc:
        best_acc = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print(str(best_acc * 100) + "%\n")

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    #print(predictions[x], x_test[x], y_test[x]); prediction; the test values; the actual value
    print("test values:", x_test[x])
    print("predicted value:", predictions[x], "// actual value:", y_test[x], "\n")

p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel(predict)
pyplot.show()