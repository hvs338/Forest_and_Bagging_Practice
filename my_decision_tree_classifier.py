from sklearn import tree
from sklearn.base import BaseEstimator

class MyDecisionTreeClassifier(BaseEstimator):

    def __init__(self):
        """
        One typically initializes shared class variables and data structures in the constructor.

        Variables which you wish to modify in train(X, y) and then utilize again in predict(X)
        should be explicitly initialized here (even only as self.my_variable = None).
        """
        self.tree = tree.DecisionTreeClassifier()

    def fit(self, X, y):
        """
        This is the method which will be called to train the model. We can assume that train will
        only be called one time for the purposes of this project.

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

        :return: self Think of this method not having a return statement at all. The idea to
        "return self" is a convention of scikit learn; the underlying model should have some
        internally saved trained state.
        """
        self.tree.fit(X,y)
        return self


        return self

    def predict(self, X):
        """
        This is the method which will be used to predict the output targets/responses of a given
        list of samples.

        It should rely on mechanisms saved after train(X, y) was called.
        You can assume that train(X, y) has already been called before this method is invoked for
        the purposes of this project.

        :param X: The samples and features which will be used for prediction. The data should have
        the shape:
        X =\
        [
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample a
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value],  # Sample b
         ...
         [feature_1_value, feature_2_value, feature_3_value, ..., feature_m_value]  # Sample n
        ]
        :return: The target/response variables the model decides is optimal for the given samples.
        The data should have the shape:
        y = [prediction_for_sample_a, prediction_for_sample_b, ..., prediction_for_sample_n]
        """
        return self.tree.predict(X)



