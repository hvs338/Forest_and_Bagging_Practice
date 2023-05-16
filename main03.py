
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.ensemble
from sklearn.datasets import make_classification

from my_random_forest_classifier import MyRandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
import time

class Experiment03:

    @staticmethod
    def run():
        """
        Loads the data, sets up the machine learning model, trains the model,
        gets predictions from the model based on unseen data, assesses the
        accuracy of the model, and prints the results.
        :return: None
        """
        X_train, X_test, y_train, y_test = Experiment03.load_data()
        my_tree = MyRandomForestClassifier()
        target_names = ["GALAXY","QSO","STAR"]
        print("[INFO]: Training my Random forest estimator")

        begin = time.time()
        my_tree.fit(X=X_train, y=y_train)
        time.sleep(1)
        end = time.time()

        print(f"[INFO]: Total time to train my Random forest estimator was {round(end - begin)} seconds")
        predictions = my_tree.predict(X=X_test)
        print(f"[INFO]: Classification report for my Random Forest estimator %")
        print(classification_report(y_test, predictions, target_names=target_names))


        sk_tree = sklearn.ensemble.RandomForestClassifier()
        print("[INFO]: Training Sklearn Random Forest")
        begin = time.time()
        sk_tree.fit(X=X_train, y=y_train)

        time.sleep(1)
        end = time.time()
        print(f"[INFO]: Total time to train my Sk-Learn Random forest was {round(end - begin)} seconds")
        sk_predictions = my_tree.predict(X=X_test)
        print(f"[INFO]: Classification report for SK-Learn Random Forest %")
        print(classification_report(y_test, sk_predictions, target_names=target_names))


    @staticmethod
    def get_accuracy(pred_y, true_y):
        """
        Calculates the overall percentage accuracy.
        :param pred_y: Predicted values
        :param true_y: Ground truth values.
        :return: The accuracy, formatted as a number in [0, 1].
        """
        if len(pred_y) != len(true_y):
            raise Exception("Different number of prediction-values than truth-values.")

        number_of_agreements = 0
        number_of_pairs = len(true_y)  # Or, equivalently, len(pred_y)

        for individual_prediction_value, individual_truth_value in zip(pred_y, true_y):
            if individual_prediction_value == individual_truth_value:
                number_of_agreements += 1

        accuracy = (number_of_agreements / number_of_pairs)*100

        return accuracy
    @staticmethod
    def load_data():

        # Extraction
        print("[INFO]: Loading Data")
        df = pd.read_csv("star_classification.csv")
        df = df[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'class', 'redshift']]
        #df = df.replace(to_replace="GALAXY",value=0)
        #df = df.replace(to_replace="QSO", value=1)
        #df = df.replace(to_replace="STAR", value=2)

        # Transform
        x = df.drop(["class"],axis=1)
        y = df["class"]
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        y = np.asarray(y)

        #Load
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


        return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    # Run the experiment once.
    Experiment03.run()