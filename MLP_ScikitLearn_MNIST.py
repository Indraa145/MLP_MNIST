from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# rescale the data, use the traditional train/test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

#Compare Training data result & original
print("For Training Data:")
print(confusion_matrix(y_train, predict_train))
print(classification_report(y_train, predict_train))

#Compare Test data result & original
print("For Test Data:")
print(confusion_matrix(y_test, predict_test))
print(classification_report(y_test, predict_test))