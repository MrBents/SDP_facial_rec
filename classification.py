from dataTranform import *
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import numpy as np
import copy
import os


class Inference:
    def __init__(self):
        if not os.path.isfile("./food.csv"):
            os.system("python dataGen.py")
        self.clf = {
            "food": {"data": DS("./food.csv")},
            "drink": {"data": DS("./drink.csv")},
            "side": {"data": DS("./side.csv")},
        }

        for key in self.clf.keys():
            self.set_classifier(self.clf[key]["data"])

    def set_classifier(self, dataset):
        X_train, X_test, y_train, y_test = dataset.split_data()
        X_train = list(X_train)
        X_train.extend(X_test)
        y_train = list(y_train)
        y_train.extend(y_test)

        clf = DecisionTreeClassifier(criterion="entropy")
        # clf = AdaBoostClassifier()
        clf.fit(X_train, y_train)
        self.clf[dataset.get_name()]["clf"] = copy.deepcopy(clf)

    # input must be in form ([age, gender, ethnicity])
    def predict(self, fpp_input):

        local = self.clf
        response = {}

        for name in local.keys():
            trans_input = [
                local[name]["data"].global_map[column]["forward"][value]
                for column, value in zip(["gender", "ethnic_origin"], fpp_input[1::])
            ]
            norm_input = [fpp_input[0], trans_input[0], trans_input[1]]

            resp = local[name]["clf"].predict([norm_input])
            response[name] = local[name]["data"].global_map[name]["reverse"][resp[0]]

        return response


inf = Inference()

# food_ds = DS("./food.csv")
# X_train, X_test, y_train, y_test = food_ds.split_data()
# X_train = list(X_train)
# X_train.extend(X_test)
# y_train = list(y_train)
# y_train.extend(y_test)

# clf = DecisionTreeClassifier(criterion="entropy")
# clf = AdaBoostClassifier(
#     base_estimator=DecisionTreeClassifier(max_depth=60), n_estimators=300
# )
# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_train)
# print(f1_score(y_train, y_pred, average="weighted"))


# drink_ds = DS("./drink.csv")
# X_train, X_test, y_train, y_test = drink_ds.split_data()
# X_train = list(X_train)
# X_train.extend(X_test)
# y_train = list(y_train)
# y_train.extend(y_test)

# clf = DecisionTreeClassifier(criterion="entropy")
# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_train)
# print(f1_score(y_train, y_pred, average="weighted"))


# side_ds = DS("./side.csv")
# X_train, X_test, y_train, y_test = side_ds.split_data()
# X_train = list(X_train)
# X_train.extend(X_test)
# y_train = list(y_train)
# y_train.extend(y_test)

# clf = DecisionTreeClassifier(criterion="entropy")
# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_train)
# print(f1_score(y_train, y_pred, average="weighted"))


# side_ds = DS("./side.csv")
# drink_ds = DS("./drink.csv")

