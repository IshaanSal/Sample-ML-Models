# ML-Models
3 Directories each working with various ML and data prediction models

The first folder: KNN explores the K Nearest Neighbors strategy. By utilizing a set of parameters associated with each grouping
of data, the KNN algorithm classifies any new data point, into one of these existing classes, based on the K number of closest
data points with respect to the new data point. Specifically, a dataset of car attributes is utilized to predict how the quality
of the car would be categorized.

The second folder: LinearRegressionModels utilizes the linear regression strategy. Given a set of data points, a linear regression model
constructs, in essence, a "line of best fit" which describes a linear line cover the general trend of the data points. This new linear
model can then be utilized to predict further results, based on a set of given parameters. Based on the overall quality of the data, the
accuracy of these models can vary. In the provided directory, a dataset of students' grades, containing dozens of attributes (e.g.
absences, failures, parent's education), as well as semester grades is provided. Utilizing a linear regression model, the program is able
to predict a student's final grade, based on the 5 parameters inputted by the user, with 90%+ accuracy.

The third folder: SVM describes the algorithm known as Support Vector Machines. In essence, this model works similar to KNN, however,
it instead creates a "hyperplane" in between the classes, to divide the data.

**Dependencies**
Numpy
Pandas
Sci Kit Learn
Matplotlib
Pickle
