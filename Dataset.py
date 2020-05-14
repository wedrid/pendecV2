import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import os


class Dataset:
    def __init__(self, name: str = "housing", directory: str = "/home/sortino/arima-exp/datasets/"):
        self.directory = directory
        self.name = name
        self.filename = self.directory
        file_list = os.listdir(directory)
        for name in file_list:
            if len(self.name.lower().split("-")[-1]) < 3:
                name_split = self.name.lower().split("-")[:-1]
            else:
                name_split = self.name.lower().split("-")
            if set(name_split) == set(os.path.splitext(name)[0].lower().split("-")):
                self.filename += name
                break

    def __drop_qms(self, data: pd.DataFrame):
        """
        Drop rows with question marks
        :param data: pandas Dataframe
        :return: filtered data
        """
        return data[(data != '?').all(axis=1)]

    def __drop_nans(self, data: pd.DataFrame):
        """
        Drop rows with NaNs
        :param data: pandas Dataframe
        :return: filtered data
        """
        return data[(data.astype(str) != 'NaN').all(axis=1)]

    def get_dataset(self):
        categorical, continuous, binary = [], [], []
        target = ""

        if "housing" in self.filename.lower():
            data = pd.read_csv(self.filename)
            data = self.__drop_qms(data)
            # X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
            # return scale(X), scale(Y)
            continuous = list(data.columns)[:-1]
            target = list(data.columns)[-1]

        elif "servo" in self.filename.lower():
            data = pd.read_csv(self.filename)
            data = self.__drop_qms(data)
            categorical = ["motor", "skew", "pgain", "vgain"]
            target = "class"

        elif "auto-mpg" in self.filename.lower():
            data = pd.read_csv(self.filename)
            data = self.__drop_qms(data)
            continuous = ["displacement", "horsepower", "weight", "acceleration"]
            categorical = ["cylinders", "model_year", "origin"]
            target = "mpg"

        elif "solar-flare" in self.filename.lower():
            data = pd.read_csv(self.filename)
            data = self.__drop_qms(data)
            categorical = ["class", "largestspotsize", "spotdistribution", "evolution", "activitycode"]
            binary = ["activity", "complexity", "historicalcomplex", "area", "arealargestspot"]
            if self.name.lower().endswith("c"):
                target = "cclass"
            elif self.name.lower().endswith("m"):
                target = "mclass"
            elif self.name.lower().endswith("x"):
                target = "xclass"
            else:
                raise NameError("[ERROR] {} has no correspondences. Please retry with a valid name.".format(self.name))

        elif "breast-cancer" in self.filename.lower():
            header = ["id", "diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean"
                      "smoothness_mean", "compactness_mean", "concavity_mean", "concave_points_mean",
                      "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se",
                      "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave_points_se",
                      "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                      "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
                      "symmetry_worst", "fractal_dimension_worst", "other1", "other2", "other3", "other4"]
            data = pd.read_csv(self.filename, names=header)
            data = self.__drop_qms(data)
            continuous = header[2:]
            # binary = ["diagnosis"]
            target = "diagnosis"

        elif "forest-fires" in self.filename.lower():
            data = pd.read_csv(self.filename)
            data = self.__drop_qms(data)
            data["XY"] = data["X"].astype(str) + data["Y"].astype(str)
            categorical = ["month", "day", "XY"]
            continuous = ["FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]
            target = "area"

        elif "automobile" in self.filename.lower():
            data = pd.read_csv(self.filename)
            data = self.__drop_qms(data)

            continuous = ["normalized-losses", "wheel-base", "length", "width", "height", "curb-weight",
                          "engine-size", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
                          "highway-mpg"]
            categorical = ["symboling", "make", "body-style", "drive-wheels", "engine-type", "num-of-cylinders",
                           "fuel-system"]
            binary = ["fuel-type", "aspiration", "num-of-doors", "engine-location"]
            target = "price"

        elif "crime" in self.filename.lower():
            data = pd.read_csv(self.filename)
            target = "ViolentCrimesPerPop"

            np_cols_to_drop = ['communityname', 'state', 'countyCode', 'communityCode', 'fold']
            target_cols_to_drop = ['murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies', 'robbbPerPop', 'assaults',
                                   'assaultPerPop', 'burglaries', 'burglPerPop', 'larcenies', 'larcPerPop', 'autoTheft',
                                   'autoTheftPerPop', 'arsons', 'arsonsPerPop', 'nonViolPerPop']
            # quasi_empty_cols_to_drop = ['NumPolice', 'policePerPop', 'policeField', 'policeFieldPerPop', 'policeCalls',
            #                             'policCallPerPop', 'policCallPerOffic', 'policePerPop2', 'racialMatch',
            #                             'pctPolicWhite', 'pctPolicBlack', 'pctPolicHisp', 'pctPolicAsian',
            #                             'pctPolicMinority', 'officDrugUnits', 'numDiffDrugsSeiz', 'policAveOT',
            #                             'policCarsAvail', 'policOperBudget', 'pctPolicPatrol', 'gangUnit',
            #                             'policBudgetPerPop']

            cols_to_drop = np_cols_to_drop + target_cols_to_drop #+ quasi_empty_cols_to_drop
            cols_to_drop = list(set(cols_to_drop))

            for c in cols_to_drop:
                if c in list(data.columns):
                    data = data.drop([c], axis=1)
                else:
                    print(c)
            for col in data.columns:
                if (data[col].astype(str) == '?').sum() > 100 and not str(col) == target:
                    data = data.drop([col], axis=1)
            data = data[data[target] != '?']
            data = self.__drop_qms(data)
            continuous = list(data.columns)
            continuous.remove(target)
        else:
            raise NameError("[ERROR] {} has no correspondences. Please retry with a valid name.".format(self.name))

        # Preparing the labels
        Y = data.pop(target)
        if target in categorical:
            Y = Y.astype('category').cat.codes
            categorical.remove(target)
        if target in binary:
            Y = pd.get_dummies(Y, drop_first=True)
            binary.remove(target)
        Y = scale(Y.values)

        # Preparing features
        if categorical:
            categorical_feats = pd.get_dummies(data[categorical].astype(str))
        else:
            categorical_feats = pd.DataFrame()

        if binary:
            binary_feats = pd.get_dummies(data[binary], drop_first=True)
        else:
            binary_feats = pd.DataFrame()

        continuous_feats = data[continuous]

        # Scaling continuous features
        if not continuous_feats.empty:
            continuous_feats = pd.DataFrame(scale(continuous_feats.values), columns=continuous)

        # Creating feature matrix
        feats = [x for x in [categorical_feats, binary_feats, continuous_feats] if not x.empty]
        X = np.concatenate(feats, axis=1)

        return X, Y
