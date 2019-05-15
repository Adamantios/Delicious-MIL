load_path = './DeliciousMIL/Data/'

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import sys
import time
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,hamming_loss,make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import sys
import time
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
import scipy.stats as sp
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV


from keras.preprocessing import *

def read_data(file,lab_file):

    X_data = pd.read_csv(file,header=None)
    y_data = pd.read_csv(lab_file,header=None)

    X_data = X_data[0].map(lambda x: re.sub('<\d+>','',x) \
        .strip() \
        .split())
    X_data = X_data.map(lambda x: [int(tok.strip()) for tok in x])
    y_data = y_data[0].map(lambda x: np.array([int(lab) for lab in x.split()]))
    
    return X_data.tolist(),np.array(y_data.tolist())







def read_data_sentences(file,lab_file,maxlen,max_sentence_len):
    
    X_data = pd.read_csv(file,header=None)
    y_data = pd.read_csv(lab_file,header=None)

    X_data = X_data[0].map(lambda x: x.strip())

    X_data = X_data.map(lambda x: re.findall('<\d+>([^<]+)',x)[1:])

    X_data = X_data.map(lambda x: [[int(tok.strip()) for tok in sent.strip().split()] for sent in x ])

    y_data = y_data[0].map(lambda x: np.array([int(lab) for lab in x.split()]))

    X_data = X_data.tolist()
    X_data_int = np.zeros((len(X_data),maxlen,max_sentence_len))
    for idx,text_bag in enumerate(X_data):
        sentences_batch = np.zeros((maxlen,max_sentence_len))
        sentences =  sequence.pad_sequences(text_bag,
            maxlen=max_sentence_len,
            padding='post',
            truncating='post',
            dtype='int32')
        for j,sent in enumerate(sentences):
            if j >= max_sentence_len:
                break
            sentences_batch[j,:] = sent
        X_data_int[idx,:,:] = sentences_batch

    X_data = X_data_int

    return X_data,np.array(y_data.tolist())







def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))







def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def load_dataset(maxlen, path='./DeliciousMIL/Data', ngram_range=1):
    """
    """
    train_data = path + '/train-data.dat'
    train_labels = path + '/train-label.dat'
    test_data = path + '/test-data.dat'
    test_labels = path + '/test-label.dat'
    vocab_file = path + '/vocabs.txt'

    print('Loading data...')
    X_train, y_train = read_data(train_data,train_labels)
    X_test, y_test = read_data(test_data,test_labels)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')
#     print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
#     print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))


    word_index = {}
    with open(vocab_file,'r') as vf:
        for line in vf:
            line = line.strip().split(', ')
            key = line[0]
            value = int(line[1])
            word_index[key] = value

    max_features = len(word_index)

    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in X_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        X_train = add_ngram(X_train, token_indice, ngram_range)
        X_test = add_ngram(X_test, token_indice, ngram_range)
#         print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
#         print('Average val sequence length: {}'.format(np.mean(list(map(len, X_val)), dtype=int)))
#         print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))


#     print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)


    return X_train,y_train,X_test,y_test,word_index


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)



def hyperparameters(classifier, params, X, y,best,scoring, clf_name, random_search=True):
    best_params = []
    cv_results = []
    best_scores = []
    print("\nΕstimator : " + clf_name)
    if random_search:
        searcher = RandomizedSearchCV(classifier, params, cv=5, n_jobs=-1,
                                      verbose=1, scoring=scoring, refit=best)
    else:
        searcher = GridSearchCV(classifier, params, cv=4, n_jobs=-1,
                                verbose=1, scoring=scoring, refit=best)

    # Finding the best parameters in the original set in order to generalize better
    searcher.fit(X, y)
    cv_results.append(searcher.cv_results_)
    best_params.append(searcher.best_params_)
    best_scores.append(searcher.best_score_)

    final_results = [best_params, cv_results, best_scores]

    print('Best parameters found for Estimator : %s' % clf_name)
    print(searcher.best_params_)
    print("\nBest score found for %s Score metric : %.3f" % (best,searcher.best_score_))

    return final_results
