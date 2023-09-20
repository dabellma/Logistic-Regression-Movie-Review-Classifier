# CS542 Fall 2021 Programming Assignment 2
# Logistic Regression Classifier

import os
import numpy as np
import re
from collections import defaultdict
from math import ceil
from random import Random

'''
Computes the logistic function.
'''
def sigma(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression():

    def __init__(self, n_features):
        # be sure to use the right class_dict for each data set
        self.class_dict = {'neg': 0, 'pos': 1}
        # self.class_dict = {'action': 0, 'comedy': 1}
        # use of self.feature_dict is optional for this assignment
        # self.feature_dict = {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}

        self.n_features = n_features
        self.theta = np.zeros(n_features + 1) # weights (and bias)

    '''
    Loads a dataset. Specifically, returns a list of filenames, and dictionaries
    of classes and documents such that:
    classes[filename] = class of the document
    documents[filename] = feature vector for the document (use self.featurize)
    '''
    def load_data(self, data_set):
        filenames = []
        classes = dict()
        documents = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # your code here
                    # BEGIN STUDENT CODE
                    filenames.append(name)


#use this code to split correctly on windows, no other files/folders in the root movie-* folder

                    c = root.split('\\')[-1]
                    classes[name] = self.class_dict[c]

###########################################


                    words = f.read().split()
                    documents[name] = self.featurize(words)
                    # END STUDENT CODE

        return filenames, classes, documents

    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''

    def featurize(self, document):
        vector = np.zeros(self.n_features + 1)
        # BEGIN STUDENT CODE

        vector[0] = self.count_combined_words(document)

        # END STUDENT CODE
        vector[-1] = 1
        return vector
    
    def count_combined_words(self, document):
        negative_words = ['dumb', 'bad', 'fucking', 'lament', 'trash', 'disappointing', 'asleep', 'sleep', 'lacks', 'boring', 'nothing', 'tired', 'worst']
        positive_words = ['refreshing', 'insightful', 'enjoyable', 'confident', 'charming', 'laugh', 'fun', 'good', 'enjoy', 'engaging']
        count = 0
        for token in document:
            if token in positive_words:
                count += 1
            if token in negative_words:
                count -= 1

        return count
    
    
    '''
    Trains a logistic regression classifier on a training set.
    '''
    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        filenames, classes, documents = self.load_data(train_set)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        for epoch in range(n_epochs):
            # print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]

                # BEGIN STUDENT CODE

                # create and fill in matrix x and vector y
                # for my x matrix, it should be len(minibatch) x self.n_features + 1 (a.k.a., (m, F + 1))
                # for my y matrix, it should be an array of len(minibatch) (a.k.a, (m,))
                x_matrix = np.zeros((len(minibatch), self.n_features + 1))
                y_vector = np.zeros(len(minibatch))
                for i in range(len(minibatch)):
                    x_matrix[i] = documents[minibatch[i]]
                    y_vector[i] = classes[minibatch[i]]

                # compute y_hat
                y_hat = sigma(np.dot(x_matrix, self.theta))

                # update loss
                loss += -(np.dot(y_vector, np.log(y_hat)) + np.dot((1 - y_vector), np.log(1 - y_hat)))

                # compute gradient
                ave_grad = (np.dot(x_matrix.transpose(), (y_hat - y_vector))) / len(minibatch)

                # update weights (and bias)
                self.theta = self.theta - (eta*ave_grad)

                # END STUDENT CODE
            loss /= len(filenames)
            # print("Average Train Loss: {}".format(loss))
            # randomize order
            Random(epoch).shuffle(filenames)

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set)
        for name in filenames:
            # BEGIN STUDENT CODE
            # get most likely class (recall that P(y=1|x) = y_hat)
            results[name]['correct'] = classes[name]

            y_hat = sigma(np.dot(documents[name], self.theta))
            if (y_hat > 0.5):
                results[name]['predicted'] = 1
            else:
                results[name]['predicted'] = 0

            # END STUDENT CODE

        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you can copy and paste your code from PA1 here

        tp_1 = 0
        tn_1 = 0
        fp_1 = 0
        fn_1 = 0
        
        tp_0 = 0
        tn_0 = 0
        fp_0 = 0
        fn_0 = 0

        for key in results:
            if ((results[key]['correct'] == 1) and (results[key]['predicted'] == 1)):
                tp_1 += 1
                tn_0 += 1
            if ((results[key]['correct'] == 0) and (results[key]['predicted'] == 0)):
                tn_1 += 1
                tp_0 += 1
            if ((results[key]['correct'] == 0) and (results[key]['predicted'] == 1)):
                fp_1 += 1
                fn_0 += 1
            if ((results[key]['correct'] == 1) and (results[key]['predicted'] == 0)):
                fn_1 += 1
                fp_0 += 1

        precision_1 = tp_1 / (tp_1 + fp_1)
        recall_1 = tp_1 / (tp_1 + fn_1)
        accuracy_1 = (tp_1 + tn_1) / (tp_1 + tn_1 + fp_1 + fn_1)
        f1_score_1 = (2*precision_1*recall_1) / (precision_1 + recall_1)

        precision_0 = tp_0 / (tp_0 + fp_0)
        recall_0 = tp_0 / (tp_0 + fn_0)
        f1_score_0 = (2*precision_0*recall_0) / (precision_0 + recall_0)


        print("Precision 1: {}, Recall 1: {}, F1 Score 1: {}".format(precision_1, recall_1, f1_score_1))
        print("Precision 0: {}, Recall 0: {}, F1 Score 0: {}".format(precision_0, recall_0, f1_score_0))
        print("Accuracy: {}%".format(100 * accuracy_1))


if __name__ == '__main__':

    lr = LogisticRegression(n_features=1)
    lr.train('movie_reviews/train', batch_size=3, n_epochs=50, eta=1E-1)
    results = lr.test('movie_reviews/dev')
    lr.evaluate(results)


    print("Starting hyperparameter tuning")
    batch_size_list = [1, 3, 10, 20]
    n_epoch_list = [10, 50, 200]
    eta_list = [1E-3, 1E-2, 1E-1]

    for bs in batch_size_list:
        for ne in n_epoch_list:
            for e in eta_list:
                print("Running with batch size: {}, n epochs: {}, eta: {}".format(bs, ne, e))

                lr = LogisticRegression(n_features=1)
                lr.train('movie_reviews/train', batch_size=bs, n_epochs=ne, eta=e)
                results = lr.test('movie_reviews/dev')
                lr.evaluate(results)


