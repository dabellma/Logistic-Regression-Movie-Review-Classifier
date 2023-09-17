# CS542 Fall 2021 Programming Assignment 2
# Logistic Regression Classifier

import os
import numpy as np
import re
from collections import defaultdict
from math import ceil
from random import Random

#TODO delete me
import textblob
import vaderSentiment

'''
Computes the logistic function.
'''
def sigma(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression():

    def __init__(self, n_features):
        # be sure to use the right class_dict for each data set
        self.class_dict = {'neg': 0, 'pos': 1}
        # use of self.feature_dict is optional for this assignment
        self.feature_dict = {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}
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

                    c = root.split('/')[-1]

                    # determine which type of dataset you are running
                    forward_slash_index = data_set.find("/")
                    data_set_type = data_set[forward_slash_index + 1:]

                    # sets the positive or negative subfolder as a binary class converted to an integer
                    if data_set_type == 'train':
                        classes[name] = self.class_dict[re.findall(r'train\\(.*)', c)[0]]
                    elif data_set_type == 'dev':
                        classes[name] = self.class_dict[re.findall(r'dev\\(.*)', c)[0]]
                    else:
                        classes[name] = self.class_dict[re.findall(r'test\\(.*)', c)[0]]                    

                    words = f.read().split()
                    documents[name] = self.featurize(words)

                    # END STUDENT CODE
        return filenames, classes, documents

    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''
    #some idea features you can use:
    #count of vowels
    #count of curse words
    #TODO consider also using feature_dict to get higher than 60.5%
    #binary word features
    def featurize(self, document):
        vector = np.zeros(self.n_features + 1)
        # BEGIN STUDENT CODE

        # vector[0] = len(document)
        # vector[2] = self.count_periods(document)
        vector[0] = self.count_negative_curse_words(document)
        vector[1] = self.count_positive_words(document)
        # vector[2] = self.count_exclamation_marks(document)


        # END STUDENT CODE
        vector[-1] = 1
        return vector

    def count_periods(self, document):
        count = 0
        for token in document:
            if token == '.':
                count += 1

        return count
    
    def count_exclamation_marks(self, document):
        count = 0
        for token in document:
            if token == '!':
                count += 1

        return count
    
    def count_negative_curse_words(self, document):
        negative_curse_words = ['fuck', 'shit', 'crap', 'bitch', 'damn', 'ass', 'asshole', 'moron', 'bastard', 'bloody', 'bullshit', 'scum', 'whore']
        count = 0
        for token in document:
            if token in negative_curse_words:
                count += 1

        return count
    
    def count_positive_words(self, document):
        positive_curse_words = ['cheerful', 'enthusiasm', 'happiness', 'confident', 'charming', 'courageous', 'ambitious', 'affection', 'delightful', 'considerate'
                                'awesome', 'vivacious', 'adventurous', 'optimism', 'admirable', 'wonderful', 'fabulous', 'lovely', 'compassion']
        count = 0
        for token in document:
            if token in positive_curse_words:
                count += 1

        return count
    



    '''
    Trains a logistic regression classifier on a training set.
    '''
    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        filenames, classes, documents = self.load_data(train_set)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            # for i in range(5), i would be 0,1,2,3,4
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
                sigma_input = np.dot(x_matrix, self.theta)
                y_hat = sigma(sigma_input)



                # update loss
                first_log_term = np.dot(y_vector, np.log(y_hat))
                second_log_term = np.dot((1 - y_vector), np.log(1 - y_hat))
                loss += -(first_log_term + second_log_term)
                # loss += -(y_vector*np.log(y_hat) + (1 - y_vector)*np.log(1 - y_hat))



                # compute gradient
                grad = (np.dot(x_matrix.transpose(), (y_hat - y_vector)))
                ave_grad = grad / len(minibatch)


                # update weights (and bias)
                self.theta = self.theta - (eta*ave_grad)

                # END STUDENT CODE
            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
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
        count_correct = 0
        for name in filenames:
            # BEGIN STUDENT CODE
            # get most likely class (recall that P(y=1|x) = y_hat)

            sigma_input = np.dot(documents[name], self.theta)
            y_hat = sigma(sigma_input)

            results[name]['correct'] = classes[name]
            if (y_hat > 0.5):
                results[name]['predicted'] = 1
            else:
                results[name]['predicted'] = 0


            if results[name]['correct'] == results[name]['predicted']:
                count_correct += 1
            # END STUDENT CODE
        print("Percentage correct: {}%".format(100 * (count_correct / len(filenames))))

        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you can copy and paste your code from PA1 here
        pass

if __name__ == '__main__':

    lr = LogisticRegression(n_features=2)
    # make sure these point to the right directories
    #lr.train('movie_reviews/train', batch_size=3, n_epochs=1, eta=0.1)
    lr.train('movie_reviews/train', batch_size=3, n_epochs=50, eta=1E-1)
    results = lr.test('movie_reviews/dev')
    # results = lr.test('movie_reviews/test')
    lr.evaluate(results)
