import pandas as pd
from sklearn.model_selection import train_test_split


class DS:
    def __init__(self, path):
        self.raw_data = pd.read_csv(path)
        self.global_map = {}
        self.data = self.discretize_data(self.raw_data)
        self.name = path.split(".")[1].split("/")[1]

    def get_name(self):
        return self.name

    def discretize_data(self, data):
        for i, column in enumerate(list(data.keys())):

            if type(data[column][0]) != str:
                continue
            counter = (i - 1) * 100
            local_map = {}
            reverse_map = {}
            for v in list(data[column].values):
                if v.lower() in local_map.keys():
                    pass
                else:
                    local_map[v.lower()] = counter
                    reverse_map[counter] = v.lower()
                    counter += 1
            self.global_map[column] = {"reverse": reverse_map, "forward": local_map}
            data[column] = data[column].apply(lambda x: local_map[x.lower()])
        return data

    def split_data(self, ts=0.33):
        X = self.data[["age", "gender", "ethnic_origin"]].values
        y = self.data.drop(columns=["age", "gender", "ethnic_origin"]).values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ts, random_state=42
        )

        return X_train, X_test, y_train, y_test


food_ds = DS("./food.csv")
side_ds = DS("./side.csv")
drink_ds = DS("./drink.csv")

