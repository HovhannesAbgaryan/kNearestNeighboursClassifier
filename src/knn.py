import numpy as np


class KNearestNeighbour(object):
    # region Summary
    """ A kNN classifier with arbitrary distance """
    # endregion Summary

    # region Constructor

    def __init__(self, X_train, y_train):
        # region Summary
        """
        Initializing the KNN object.
        :param X_train: A numpy array or pandas DataFrame of shape (num_train, D)
        :param y_train: A numpy array or pandas Series of shape (N) containing the training labels
        """
        # endregion Summary

        self.X_train = X_train
        self.y_train = y_train

    # endregion Constructor

    # region Functions

    def compute_distances(self, X_test):
        # region Summary
        """
        Compute the distance between each test point in X_test and each training point in self.X_train using a nested
        loop over both the training data and the test data.
        :param X_test: A numpy array or pandas DataFrame of shape (num_test, D) containing test data.
        :return: dists: A numpy array of shape (num_test, num_train) where dists[i, j] is the Hamming distance between
                        the ith test point and the jth training point.
        """
        # endregion Summary

        # region Body

        X_train = self.X_train
        num_train = X_train.shape[0]
        num_test = X_test.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = (X_train != X_test.iloc[i, :]).sum(axis=1)
        return dists

        # endregion Body

    def predict_labels(self, dists, k=1):
        # region Summary
        """
        Given a matrix of distances between test points and training points, predict a label for each test point.
        :param dists: A numpy array of shape (num_test, num_train) where dists[i, j] gives the distance between the
                      ith test point and the jth training point.
        :param k: The number of nearest neighbours
        :return: A numpy array or pandas Series of shape (num_test,) containing the predicted labels for the test data.
        """
        # endregion Summary

        def majority_class(y):
            return y.value_counts().index[0]

        # region Body

        if k == 1:
            y_pred = self.y_train.iloc[dists.argmin(axis=1)]
        else:
            y_pred = []
            for i in range(dists.shape[0]):
                closest_k = np.argsort(dists[i, :])[:k]
                closest_y = self.y_train.iloc[closest_k]
                y_pred.append(majority_class(closest_y))

        return np.array(y_pred)

        # endregion Body

    def fit_predict(self, X_test, k=1):
        # region Summary
        """
        This method fits the data and predicts the labels for the given test data.
        For k-nearest neighbours fitting (training) is just memorizing the training data.
        :param X_test: A numpy array or pandas DataFrame of shape (num_test, D)
        :param k: The number of nearest neighbors.
        :return: A numpy array or pandas Series of shape (num_test,) containing predicted labels
        """
        # endregion  Summary

        # region Body

        dists = self.compute_distances(X_test)
        return self.predict_labels(dists, k=k)

        # endregion Body

    # endregion Functions
