import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data", sep=',')
print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
print(buying)

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
print(x_train, y_test)

neighbors = 1
counter = 0
best_acc = 0
optimal_n = 0
while neighbors < 15:
    for counter in range(30):
        model = KNeighborsClassifier(n_neighbors=neighbors)
        model.fit(x_train, y_train)
        acc = model.score(x_test, y_test)

        if acc > best_acc:
            best_acc = acc
            optimal_n = neighbors
    neighbors += 1

print("\noptimal neighbors: ", optimal_n)
print(str(best_acc * 100) + "%\n")

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", y_test[x])