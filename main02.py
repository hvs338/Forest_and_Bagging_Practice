from my_decision_tree_classifier import MyDecisionTreeClassifier
from main01 import Experiment01
from sklearn.model_selection import KFold
import numpy as np
from visualize import plot_training

class Experiment02:

    @staticmethod
    def run():
        """
        Loads the data, sets up the machine learning model, trains the model,
        gets predictions from the model based on unseen data, assesses the
        accuracy of the model, and prints the results.
        :return: None
        """
        number_of_folds = 5

        X_data, y_data = Experiment02._load_data()
        folds = Experiment02.get_folds(k=number_of_folds, X=X_data, y=y_data)

        ## Singular decision Tree

        train_X, train_y, test_X, test_y = Experiment01.load_data()
        my_tree = MyDecisionTreeClassifier()

        my_tree.fit(X=train_X, y=train_y)
        predictions = my_tree.predict(X=test_X)

        plot_training(my_tree,train_X,train_y,"Single Decision Tree with whole data set")
        print("Accuracy for Single Decision Tree with whole data set:")
        accuracy = Experiment01._get_accuracy(predictions,test_y)
        print(accuracy)


        for fold_index, (train_X, train_y, test_X, test_y) in enumerate(folds):
            my_tree = MyDecisionTreeClassifier()

            my_tree.fit(X=train_X, y=train_y)
            predictions = my_tree.predict(X=test_X)
            accuracy = Experiment02.get_accuracy(my_tree,predictions, test_y)

            plot_training(my_tree,train_X,train_y, "fold"+str(fold_index))

            print("Accuracy on fold ", fold_index, "is", accuracy)

    @staticmethod
    def get_accuracy(model,pred_y, true_y):
        """
        Calculates the overall percentage accuracy.
        :param pred_y: Predicted values.
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
    def get_folds(k, X, y):
        """
        Partition the data into k different folds.
        :param k: The number of folds
        :param X: The samples and features which will be used for training. The data should have
        the shape:
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]
        :param y: The target/response variable used for training. The data should have the shape:
        y = [target_for_sample_a, target_for_sample_b, ..., target_for_sample_n]
        :return: A list of k folds of the form [fold_1, fold_2, ..., fold_k]
        Each fold should be a tuple of the form
        fold_i = (train_X, train_y, test_X, test_y)
        """

        folds = []
        kf = KFold(n_splits=k,random_state = 1, shuffle= True)

        for train_index,test_index in kf.split(X):

            train_x, test_x = X[train_index,:],X[test_index,:]
            train_y, test_y = y[train_index],y[test_index]
            fold = (train_x, train_y, test_x, test_y)

            folds.append(fold)
        return folds
    @staticmethod



    @staticmethod
    def _load_data(filename="path/to/iris_data.csv"):
        """
        Load the data, separating it into a list of samples and their corresponding outputs
        :param filename: The location of the data to load from file.
        :return: X, y; each as an iterable object(like a list or a numpy array).
        The data should have
        the shape:
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]

        y = [target_for_sample_a, target_for_sample_b, ..., target_for_sample_n]
        """

        # Modify anything in this method, but keep the return line the same.
        # You may also import any needed library (like numpy)


        data = np.loadtxt("iris_data.csv",delimiter=",",skiprows=1)
        X = data[:,[1,2]]
        y = data[:,0]

        return X, y


if __name__ == "__main__":
    # Run the experiment once.
    Experiment02.run()
