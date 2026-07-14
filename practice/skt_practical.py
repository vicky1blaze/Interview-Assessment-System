from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

X = [
    [2, 5],
    [4, 7],
    [6, 8],
    [1, 4],
    [5, 6]
]

y = [
    40,
    60,
    80,
    30,
    70
]

Y = [
    40,
    60,
    10,
    30,
    50
]

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42
# )

# print("Trainig: ", X_train)
# print("Testing: ", X_test)

# model = LinearRegression()

# model.fit(X_train, y_train)
# print("Prediction: ", model.predict(X_test))

accuracy = accuracy_score(y,Y)
print(accuracy * 100)