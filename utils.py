import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sp
from keras.preprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from skmultilearn.ensemble import RakelD
from skmultilearn.problem_transform import BinaryRelevance
from termcolor import colored

labels = ['programming', 'style', 'reference', 'java', 'web', 'internet', 'culture', 'design', 'education', 'language',
          'books', 'writing', 'computer', 'english', 'politics', 'history', 'philosophy', 'science', 'religion',
          'grammar']


def read_data(file, lab_file):
    X_data = pd.read_csv(file, header=None)
    y_data = pd.read_csv(lab_file, header=None)
    X_data = X_data[0].map(lambda x: re.sub('<\d+>', '', x).strip().split())
    X_data = X_data.map(lambda x: [int(tok.strip()) for tok in x])
    y_data = y_data[0].map(lambda x: np.array([int(lab) for lab in x.split()]))

    return X_data.tolist(), np.array(y_data.tolist())


def load_dataset(maxlen, path='./DeliciousMIL/Data', binary=False):
    train_data = path + '/train-data.dat'
    train_labels = path + '/train-label.dat'
    test_data = path + '/test-data.dat'
    test_labels = path + '/test-label.dat'

    print('Loading data...')
    X_train, y_train = read_data(train_data, train_labels)
    X_test, y_test = read_data(test_data, test_labels)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    # Padding data.
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    if binary:
        # Get the most frequent class only.
        print('\nGetting the most frequent class...')
        most_frequent_counts = np.sum(np.transpose(y_train), axis=1)
        most_frequent_index = most_frequent_counts.argmax()
        y_train = y_train[:, most_frequent_index]
        y_test = y_test[:, most_frequent_index]
        print('The most frequent class was the word \'{}\', with {} appearances.'
              .format(labels[most_frequent_index], most_frequent_counts.max()))

    return X_train, y_train, X_test, y_test


def redefine(base, keys, values):
    """
    Inputs a dictionary keys and values and a base string
    and outputs a new dictionary with the base string concatenated.
    """
    new_k = [base + keys[i] for i in range(len(keys))]
    dictionary = {new_k[i]: values[i] for i in range(len(keys))}
    return dictionary


def micro_prec(y_true, y_pred):
    return precision_score(y_true, y_pred, average='micro')


def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def pipeline(method, X_train, y_train, scoring, params=None, search_r=True, best=None):
    if search_r:
        # Random search params
        r = np.random.uniform(-2, 2, size=5)
        C = np.array(10 ** r)
        alpha = np.random.uniform(0, 1, size=5)

        params_tree = {'__max_depth': sp.randint(1, 30),
                       '__max_features': sp.randint(1, X_train.shape[1]),
                       '__min_samples_split': sp.randint(2, X_train.shape[0] / 3),
                       '__criterion': ['gini', 'entropy']}
        params_lgr = {'__C': C}
        params_nb = {'__alpha': alpha}

        tree_k, tree_v = list(params_tree.keys()), list(params_tree.values())
        lgr_k, lgr_v = list(params_lgr.keys()), list(params_lgr.values())
        nb_k, nb_v = list(params_nb.keys()), list(params_nb.values())

    else:
        params_cc, params_rk, params_bn = params[0], params[1], params[2]

    if method == 'CC':
        base_str = 'base_estimator'
        if search_r:
            params_tree, params_lgr, params_nb = redefine(base_str, tree_k, tree_v), \
                                                 redefine(base_str, lgr_k, lgr_v), \
                                                 redefine(base_str, nb_k, nb_v)
            params = [params_lgr, params_tree, params_nb]
        else:
            params = params_cc
            tree_k, tree_v = list(params[1].keys()), list(params[1].values())
            lgr_k, lgr_v = list(params[0].keys()), list(params[0].values())
            nb_k, nb_v = list(params[2].keys()), list(params[2].values())

            params_tree, params_lgr, params_nb = redefine(base_str, tree_k, tree_v), \
                                                 redefine(base_str, lgr_k, lgr_v), \
                                                 redefine(base_str, nb_k, nb_v)
            params = [params_lgr, params_tree, params_nb]

        print(colored('Fitting Classifiers Chain pipeline...', 'green'))
        classifiers = {
            "Logistic Regression": ClassifierChain(LogisticRegression(random_state=0, solver='lbfgs', n_jobs=-1)),
            "Decision Tree Classifier": ClassifierChain(DecisionTreeClassifier()),
            "MultinomialNB": ClassifierChain(MultinomialNB())}

    elif method == 'RAkEL':
        base_str = 'base_classifier'
        if search_r:
            params_tree, params_lgr, params_nb = redefine(base_str, tree_k, tree_v), \
                                                 redefine(base_str, lgr_k, lgr_v), \
                                                 redefine(base_str, nb_k, nb_v)
            params = [params_lgr, params_tree, params_nb]

        else:
            params = params_rk
            tree_k, tree_v = list(params[1].keys()), list(params[1].values())
            lgr_k, lgr_v = list(params[0].keys()), list(params[0].values())
            nb_k, nb_v = list(params[2].keys()), list(params[2].values())

            params_tree, params_lgr, params_nb = redefine(base_str, tree_k, tree_v), \
                                                 redefine(base_str, lgr_k, lgr_v), \
                                                 redefine(base_str, nb_k, nb_v)
            params = [params_lgr, params_tree, params_nb]
        print(colored('Fitting RAkEL pipeline...', 'green'))
        classifiers = {"Logistic Regression": RakelD(LogisticRegression(random_state=0, solver='lbfgs', n_jobs=-1)),
                       "Decision Tree Classifier": RakelD(DecisionTreeClassifier(),
                                                          labelset_size=5),
                       "MultinomialNB": RakelD(MultinomialNB(),
                                               labelset_size=5)}

    elif method == 'BinaryRelevance':
        base_str = 'classifier'
        if search_r:

            params_tree, params_lgr, params_nb = redefine(base_str, tree_k, tree_v), \
                                                 redefine(base_str, lgr_k, lgr_v), \
                                                 redefine(base_str, nb_k, nb_v)
            params = [params_lgr, params_tree, params_nb]
        else:
            params = params_bn
            tree_k, tree_v = list(params[1].keys()), list(params[1].values())
            lgr_k, lgr_v = list(params[0].keys()), list(params[0].values())
            nb_k, nb_v = list(params[2].keys()), list(params[2].values())

            params_tree, params_lgr, params_nb = redefine(base_str, tree_k, tree_v), \
                                                 redefine(base_str, lgr_k, lgr_v), \
                                                 redefine(base_str, nb_k, nb_v)
            params = [params_lgr, params_tree, params_nb]
        print(colored('Fitting BinaryRelevance pipeline...', 'green'))
        classifiers = {
            "Logistic Regression": BinaryRelevance(LogisticRegression(random_state=0, solver='lbfgs', n_jobs=-1)),
            "Decision Tree Classifier": BinaryRelevance(DecisionTreeClassifier()),
            "MultinomialNB": BinaryRelevance(MultinomialNB())}

    else:
        raise ValueError('Invalid method passed. Expected one of: "CC", "RAkEL", "BinaryRelevance", got {} instead'
                         .format(method))

    res = {}
    for keys, classifier, par in zip(classifiers.keys(), classifiers.values(), params):
        res[keys] = hyperparameters_search(classifier, par, X_train, y_train, best, scoring, keys,
                                           candidates=30, random_search=search_r)


def hyperparameters_search(classifier, params, X, y, best, scoring, clf_name, candidates=10, cv=10, random_search=True,
                           verbose=1):
    best_params = []
    cv_results = []
    best_scores = []
    print('\n' + colored('Estimator: ', 'blue') + clf_name)
    if random_search:
        searcher = RandomizedSearchCV(classifier, params, n_iter=candidates, cv=cv, n_jobs=-1,
                                      verbose=verbose, scoring=scoring, refit=best)
    else:
        searcher = GridSearchCV(classifier, params, cv=4, n_jobs=-1,
                                verbose=verbose, scoring=scoring, refit=best)

    # Finding the best parameters in the original set in order to generalize better
    searcher.fit(X, y)
    cv_results.append(searcher.cv_results_)
    best_params.append(searcher.best_params_)
    best_scores.append(searcher.best_score_)

    results = [best_params, cv_results, best_scores]

    print('Best parameters found for Estimator : %s' % clf_name)
    print(searcher.best_params_)
    print("\nBest score found for %s Score metric : %.3f" % (best, searcher.best_score_))

    return results


def scores(name, y_test, y_pred):
    if name == 'acc':
        return accuracy_score(y_test, y_pred)
    elif name == 'hamming_loss':
        return hamming_loss(y_test, y_pred)
    elif name == 'f1_micro':
        return f1_score(y_test, y_pred, average='micro')
    elif name == 'f1_macro':
        return f1_score(y_test, y_pred, average='macro')
    elif name == 'prec_micro':
        return precision_score(y_test, y_pred, average='micro')
    elif name == 'prec_macro':
        return precision_score(y_test, y_pred, average='macro')


def plot(metrics, clf, k, steps, c):
    plt.figure(figsize=(10, 8))
    plt.subplot(k)
    plt.plot(steps, metrics[clf][2], c=c, label='RAkEL')  # 2 because we want the f1 micro score
    plt.legend()
    plt.ylim(0.05, 0.62)
    plt.title('RAkEL ' + clf + 'MicroF1 score')
    plt.ylabel('Test MicroF1')
    plt.xlabel('labelset size')
    plt.grid(True)
    print('Maximum MicroF1 RAkEL ' + clf, np.round(max(metrics[clf][2]), 4))


def RAkEL_plots(steps, metrics):
    colors = ['r', 'g', 'b']
    names = ['DecisionTreeClassifier', 'Logistic Regression', 'MultinomialNB']
    for n, k, c in zip(names, range(311, 314), colors):
        plot(metrics, n, k, steps, c)

    plt.show()


def RAkEL_fit(clfs, steps, X_train, y_train, X_test, y_test):
    metrics = {}
    for key, clf in zip(clfs.keys(), clfs.values()):
        acc = []
        prec_micro = []
        prec_macro = []
        hamm_loss = []
        f1_micro = []
        f1_macro = []
        print('Fitting RAkEL with Base Classifier: %s' % key)
        for k in steps:
            classifier = RakelD(base_classifier=clf, labelset_size=k)
            classifier.fit(X_train, y_train)
            prediction = classifier.predict(X_test)
            acc.append(accuracy_score(y_test, prediction))
            prec_micro.append(precision_score(y_test, prediction, average='micro'))
            prec_macro.append(precision_score(y_test, prediction, average='macro'))
            hamm_loss.append(hamming_loss(y_test, prediction))
            f1_micro.append(f1_score(y_test, prediction, average='micro'))
            f1_macro.append(f1_score(y_test, prediction, average='macro'))

        metrics[key] = [acc, hamm_loss, f1_micro, f1_macro, prec_micro, prec_macro]

    return metrics


def CC_Fit(clfs, X_train, y_train, X_test, y_test, evaluate):
    metrics_cc = {}
    for key, clf in zip(clfs.keys(), clfs.values()):
        print('Fitting Chain %s' % key)
        chains = [ClassifierChain(clf, order='random', random_state=i) for i in range(10)]
        for chain in chains:
            chain.fit(X_train, y_train)

        Y_pred_chains = np.array([chain.predict(X_test) for chain in
                                  chains])

        pred_ens = Y_pred_chains.mean(axis=0)
        # Chain scores

        for m in evaluate:
            metrics_cc[key + ' ' + m] = [scores(m, y_test, y_pred >= .5) for y_pred in Y_pred_chains]
            metrics_cc[key + ' ' + m + ' ensemble'] = scores(m, y_test, pred_ens >= .5)
    return metrics_cc


def CC_plots(model_names, clfs, metrics_cc, ind_scores):
    for key in clfs.keys():
        loss = [ind_scores[key + ' ' + 'f1_micro']]
        loss += metrics_cc[key + ' ' + 'f1_micro']
        loss.append(metrics_cc[key + ' ' + 'f1_micro' + ' ensemble'])

        x_pos = np.arange(len(model_names))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.grid(True)
        ax.set_title('Classifier Chain ' + key + ' Ensemble Performance Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation='vertical')
        ax.set_ylabel('Hamming Loss')
        ax.set_ylim([min(loss) * .9, max(loss) * 1.1])
        colors = ['r'] + ['b'] * (len(model_names) - 2) + ['g']
        ax.bar(x_pos, loss, alpha=0.5, color=colors)
        plt.tight_layout()
        print('Maximum F1 score Classifier Chain ' + key, np.round(max(loss), 4))
        plt.show()


def BN_fit(clfs, X_train, y_train, X_test, y_test, evaluate):
    metrics_lb = {}
    for key, clf in zip(clfs.keys(), clfs.values()):
        print('Fitting BinaryRelevance with Classifier : %s' % key)
        clf = BinaryRelevance(clf)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        for m in evaluate:
            metrics_lb[key + ' ' + m] = scores(m, y_test, preds)
    return metrics_lb


def BN_plots(scores_list, names):
    fig, ax = plt.subplots()
    ind = np.arange(1, 4)
    # show the figure, but do not block
    pm, pc, pn = plt.bar(ind, scores_list)
    pm.set_facecolor('r')
    pc.set_facecolor('g')
    pn.set_facecolor('b')
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    ax.set_ylim([0.05, 0.50])
    ax.set_ylabel('Hamming Loss Score')
    plt.show()
    print('Maximum Score with Label Powersets', np.round(np.max(scores_list), 4), names[np.argmax(scores_list)])


def final_results(metrics_cc, metrics_rk, metrics_lb, evaluate, names):
    names_res = ['CC', 'RakEL', 'LabelPowerset']
    scores_rk = {}
    for n in names:
        for m, k in zip(evaluate, range(len(evaluate))):
            scores_rk[n + ' ' + m] = metrics_rk[n][k][0]

    res = [metrics_cc, scores_rk, metrics_lb]

    df = {}
    for n in names:
        for r, k in zip(names_res, res):
            for m in evaluate:
                if r == 'CC':
                    df[n + ' ' + r + ' ' + m] = k[n + ' ' + m + ' ' + 'ensemble']
                else:
                    df[n + ' ' + r + ' ' + m] = k[n + ' ' + m]

    df_lr = {}
    df_dt = {}
    df_nb = {}
    for n_s in names_res:
        for m in evaluate:
            df_lr[n_s + ' ' + m] = df['Logistic Regression' + ' ' + n_s + ' ' + m]
            df_dt[n_s + ' ' + m] = df['DecisionTreeClassifier' + ' ' + n_s + ' ' + m]
            df_nb[n_s + ' ' + m] = df['MultinomialNB' + ' ' + n_s + ' ' + m]

    data_lr = [{'acc': df_lr[i + ' acc'], 'hamming_loss': df_lr[i + ' hamming_loss'],
                'f1_micro': df_lr[i + ' f1_micro'], 'f1_macro': df_lr[i + ' f1_macro'],
                'prec_micro': df_lr[i + ' prec_micro'], 'prec_macro': df_lr[i + ' prec_macro']} for i in names_res]
    data_dt = [{'acc': df_dt[i + ' acc'], 'hamming_loss': df_dt[i + ' hamming_loss'],
                'f1_micro': df_dt[i + ' f1_micro'], 'f1_macro': df_dt[i + ' f1_macro'],
                'prec_micro': df_dt[i + ' prec_micro'], 'prec_macro': df_dt[i + ' prec_macro']} for i in names_res]
    data_nb = [{'acc': df_nb[i + ' acc'], 'hamming_loss': df_nb[i + ' hamming_loss'],
                'f1_micro': df_nb[i + ' f1_micro'], 'f1_macro': df_nb[i + ' f1_macro'],
                'prec_micro': df_nb[i + ' prec_micro'], 'prec_macro': df_nb[i + ' prec_macro']} for i in names_res]

    pd1 = pd.DataFrame(data_lr)
    pd2 = pd.DataFrame(data_dt)
    pd3 = pd.DataFrame(data_nb)

    pd1 = pd1.rename(index={0: 'Logistic Regression CC', 1: 'Logistic Regression RAkEL',
                            2: 'Logistic Regression Binary Relevance'})
    pd2 = pd2.rename(index={0: 'DecisionTreeClassifier CC', 1: 'DecisionTreeClassifier RAkEL',
                            2: 'DecisionTreeClassifier Binary Relevance'})
    pd3 = pd3.rename(index={0: 'MultinomialNB CC', 1: 'MultinomialNB RAkEL',
                            2: 'MultinomialNB Binary Relevance'})

    frames = [pd1, pd2, pd3]
    pdfinal = pd.concat(frames)

    return pdfinal


def find_in_dict(target, d):
    for key, value in d.items():
        if value == target:
            return key


def best_results(final_res):
    for key in final_res.keys():
        if key == 'hamming_loss':
            score = np.min(final_res[key].values)
        else:
            score = np.max(final_res[key].values)
        print('Best %s found : %.4f with classifier and method :%s' % (key, score, find_in_dict(score, final_res[key])))
