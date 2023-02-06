import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

print(np.__version__)
print(pd.__version__)
print(sklearn.__version__)

p1 = input("Input parameter 1:")
p2 = input("Input parameter 2:")
p3 = input("Input parameter 3:")
p4 = input("Input parameter 4:")
p5 = input("Input parameter 5:")

data = pd.read_csv("student-mat.csv", sep=";") #identifies the file being scanned as well as the character separating the data
data = data[[p1, p2, p3, p4, p5, "G3"]] #the sample data being identified

predict = "G3"
x = np.array(data.drop([predict], 1)) #attributes (uses predict because this allows you to change the attribute being looked at)
y = np.array(data[predict]) #labels

best_acc = 0
for counter in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best_acc:
        best_acc = acc

print(str(best_acc * 100) + "%\n")

print("Co: " + str(linear.coef_))
print("Intercept: " + str(linear.intercept_) + "\n")

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    #print(predictions[x], x_test[x], y_test[x]); prediction; the test values; the actual value
    print("test values:", x_test[x])
    print("predicted value:", predictions[x], "// actual value:", y_test[x], "\n")