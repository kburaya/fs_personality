import logging
import math
import utils
import uuid
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn import cross_validation
from sklearn.decomposition import NMF
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

global users, text, media, location, text_media, text_location, media_location, text_media_location




def build_x_y(dataset, label):
    try:
        x = dataset.drop(['_id', 'Foursquare', 'Instagram', 'Twitter', 'gender', 'mbti'], axis=1)
    except:
        print('Failed in deleting unxestising fields, skipping')
        x = dataset

    if label == 'e_i':
        x = x.drop(['s_n', 't_f', 'j_p'], axis=1)
    elif label == 's_n':
        x = x.drop(['e_i', 't_f', 'j_p'], axis=1)
    elif label == 't_f':
        x = x.drop(['e_i', 's_n', 'j_p'], axis=1)
    elif label == 'j_p':
        x = x.drop(['e_i', 's_n', 't_f'], axis=1)
    x = x.dropna()
    y = x[label]
    x = x.drop(label, axis=1)

    if y.value_counts().values[0] > y.value_counts().values[1]:
        print('%s is positive label, %s is negative label' %
              (y.value_counts().index[0], y.value_counts().index[1]))
        y[y == y.value_counts().index[0]] = True
        y[y == y.value_counts().index[1]] = False

    else:
        print('%s is positive label, %s is negative label' %
              (y.value_counts().index[1], y.value_counts().index[0]))
        y[y == y.value_counts().index[1]] = True
        y[y == y.value_counts().index[0]] = False
    y = list(y)

    return x, y


def get_groundtruth(train_output, test_output, label_position):
    zero_label_count, one_label_count = 0, 0
    binary_train_output, binary_test_output = list(), list()
    for label_string in train_output:
        if label_string[label_position] in utils.labels_zero:
            binary_train_output.append(0)
            zero_label_count += 1
        else:
            binary_train_output.append(1)
            one_label_count += 1

    zero_label_count, one_label_count = 0, 0
    for label_string in test_output:
        if label_string[label_position] in utils.labels_zero:
            binary_test_output.append(0)
            zero_label_count += 1
        else:
            binary_test_output.append(1)
            one_label_count += 1

    return binary_train_output, binary_test_output


def modelfit(alg, x, y, performCV=True, printFeatureImportance=False, cv_folds=5, scoring='accuracy'):
    # Fit the algorithm on the data
    alg.fit(x, y)

    # Predict training set:
    predictions = alg.predict(x)
    predprob = alg.predict_proba(x)[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, x, y, cv=cv_folds, scoring=scoring)

    if performCV:
        logging.info(
            "CV %s Score : Mean - %.3g | Std - %.3g | Min - %.3g | Max - %.3g" % (
                scoring, np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))

        # Print Feature Importance:
        # if printFeatureImportance:
        #     feat_imp = pd.Series(alg.feature_importances_, x.columns).sort_values(ascending=False)
        #     feat_imp = feat_imp[:15]
        #     feat_imp.plot(kind='bar', title='Feature Importances')
        #     plt.ylabel('Feature Importance Score')


# read_data()
# read_and_fill_missed_data()

# data_sources_names = ['text', 'media', 'location', 'text_media', 'text_location', 'media_location', 'text_media_location']
# data_sources = [text, media, location, text_media, text_location, media_location, text_media_location]
labels = ['e_i', 's_n', 't_f', 'j_p']
train_input, train_output, test_input, test_output = utils.fill_missed_modality('full_text_media_location')


def fit_gbm_results():
    logging.basicConfig(filename='gbm.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.info('GBM model')
    score = 'f1'
    for label, label_position in zip(labels, range(0, 4)):
        logging.info("Label [%s]" % label)
        # preprocess default values
        x_train, x_test = train_input, test_input
        y_train, y_test = get_groundtruth(train_output, test_output, label_position)
        default_samples_split = int(len(y_train) * 0.01)
        default_number_of_features = int(math.sqrt(len(x_train[0])))
        # 1st step, fitting learning_rate, n_estimators
        param_test1 = {'n_estimators': range(20, 121, 10),
                       'learning_rate': [0.001, 0.01, 0.05, 0.1]}
        gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(min_samples_split=default_samples_split,
                                                                     min_samples_leaf=50,
                                                                     max_depth=8,
                                                                     max_features='sqrt',
                                                                     subsample=0.8,
                                                                     random_state=10),
                                param_grid=param_test1,
                                scoring=score,
                                n_jobs=10,
                                iid=False,
                                cv=5)

        gsearch1.fit(x_train, y_train)
        test_set_score = gsearch1.score(x_test, y_test)

        best_learning_rate = gsearch1.best_params_['learning_rate']
        best_n_estimators = gsearch1.best_params_['n_estimators']
        best_score = gsearch1.best_score_
        logging.info("learning_rate: [%.3g], n_estimators: [%d]" % (best_learning_rate, best_n_estimators))
        logging.info("local best on train_set: [%.3g]" % best_score)
        logging.info("score on test_set: [%.3g]" % test_set_score)

        log_results(x_test, y_test, gsearch1, label_position)

        # 2nd step, fitting max_depth, min_samples_split
        param_test2 = {'max_depth': range(5, 16, 1), 'min_samples_split': range(25, 551, 25)}
        gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=best_learning_rate,
                                                                     n_estimators=best_n_estimators,
                                                                     max_features='sqrt',
                                                                     subsample=0.8,
                                                                     random_state=10),
                                param_grid=param_test2, scoring=score, n_jobs=10, iid=False, cv=5)
        gsearch2.fit(x_train, y_train)
        test_set_score = gsearch2.score(x_test, y_test)
        best_max_depth = gsearch2.best_params_['max_depth']
        best_min_samples_split = gsearch2.best_params_['min_samples_split']
        best_score = gsearch2.best_score_
        logging.info("max_depth: [%d], min_samples_split: [%d]" % (best_max_depth, best_min_samples_split))
        logging.info("local best score: [%.3g]" % best_score)
        logging.info("score on test_set: [%.3g]" % test_set_score)

        log_results(x_test, y_test, gsearch1, label_position)

        # 3d step, fitting min_samples_split, min_samples_leaf
        param_test3 = {'min_samples_split': range(50, 551, 50), 'min_samples_leaf': range(1, 101, 10)}
        gsearch3 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=best_learning_rate,
                                                                     n_estimators=best_n_estimators,
                                                                     max_depth=best_max_depth,
                                                                     max_features='sqrt',
                                                                     subsample=0.8,
                                                                     random_state=10),
                                param_grid=param_test3, scoring=score, n_jobs=10, iid=False, cv=5)
        gsearch3.fit(x_train, y_train)
        test_set_score = gsearch3.score(x_test, y_test)
        best_min_samples_split = gsearch3.best_params_['min_samples_split']
        best_min_samples_leaf = gsearch3.best_params_['min_samples_leaf']
        best_score = gsearch3.best_score_
        logging.info(
            "min_samples_split: [%d], min_samples_leaf: [%d]" % (best_min_samples_split, best_min_samples_leaf))
        logging.info("local best score: [%.3g]" % best_score)
        logging.info("score on test_set: [%.3g]" % test_set_score)

        log_results(x_test, y_test, gsearch1, label_position)

        # 4th step, fitting max_features
        min_features = int(default_number_of_features - 0.6 * default_number_of_features)
        max_features = int(default_number_of_features + 0.6 * default_number_of_features)
        param_test4 = {'max_features': range(min_features, max_features, 1)}
        gsearch4 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=best_learning_rate,
                                                                     n_estimators=best_n_estimators,
                                                                     max_depth=best_max_depth,
                                                                     min_samples_split=best_min_samples_split,
                                                                     min_samples_leaf=best_min_samples_split,
                                                                     subsample=0.8,
                                                                     random_state=10),
                                param_grid=param_test4, scoring=score, n_jobs=8, iid=False, cv=5)
        gsearch4.fit(x_train, y_train)
        test_set_score = gsearch4.score(x_test, y_test)
        best_max_features = gsearch4.best_params_['max_features']
        best_score = gsearch4.best_score_
        logging.info("max_features: [%d]" % best_max_features)
        logging.info("local best score: [%.3g]" % best_score)
        logging.info("score on test_set: [%.3g]" % test_set_score)

        log_results(x_test, y_test, gsearch1, label_position)
        joblib.dump(gsearch4.best_estimator_, 'models/gbm_%d.pkl' % label_position, compress=1)


def fit_bayes_model_results():
    logging.basicConfig(filename='bayes.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.info('Bayes model')
    for label, label_position in zip(labels, range(0, 4)):
        logging.info("Label [%s]" % label)
        # preprocess default values
        x_train, x_test = train_input, test_input
        y_train, y_test = get_groundtruth(train_output, test_output, label_position)
        NB = GaussianNB()
        NB.fit(x_train, y_train)

        accuracy = NB.score(x_test, y_test)
        logging.info('Average accuracy %.2f' % accuracy)
        log_results(x_test, y_test, NB.predict_proba(x_test), NB, label_position)

        joblib.dump(NB, 'models/nb_%d.pkl' % label_position, compress=1)


def fit_logistic_regression():
    logging.basicConfig(filename='lg.log', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.info('LG model')
    for label, label_position in zip(labels, range(0, 4)):
        if label_position == 0 or label_position == 1:
            continue
        logging.info("Label [%s]" % label)
        # preprocess default values
        x_train, x_test = train_input, test_input
        y_train, y_test = get_groundtruth(train_output, test_output, label_position)
        params = {'C': [0.1, 0.5, 1.0],
                  'penalty': ['l1', 'l2']}
        lg_model = LogisticRegression()
        logging.info('Begin to train model...')
        lg_gridsearch = GridSearchCV(estimator=lg_model, param_grid=params, cv=5, n_jobs=4, scoring='f1')
        lg_gridsearch.fit(x_train, y_train)
        log_results(x_test, y_test, lg_gridsearch.predict_proba(x_test), lg_gridsearch, label_position)

        logging.info('Best params for Logistic Regression: C=%.2f, regularization=%s' %
                     (lg_gridsearch.best_params_['C'], lg_gridsearch.best_params_['penalty']))

        joblib.dump(lg_gridsearch.best_estimator_, 'models/lg_%d.pkl' % label_position, compress=1)


def log_results(x_test, y_test, y_proba, model, label_position):
    # zero label
    precision = precision_score(y_test, model.predict(x_test), pos_label=0)
    recall = recall_score(y_test, model.predict(x_test), pos_label=0)
    f_score = f1_score(y_test, model.predict(x_test), pos_label=0)
    f_score_macro = f_score
    if y_proba is not None:
        fp_rate, tp_rate, thresholds = roc_curve(y_test, y_proba[:, 0], pos_label=0)
        auc_score = auc(fp_rate, tp_rate)
    else:
        auc_score = 'not defined'
    logging.info('For label %s: precision %.2f, recall %.2f, f_score %.2f, auc %s' %
                 (labels_zero[label_position], precision, recall, f_score, str(auc_score)))

    # one label
    precision = precision_score(y_test, model.predict(x_test), pos_label=1)
    recall = recall_score(y_test, model.predict(x_test), pos_label=1)
    f_score = f1_score(y_test, model.predict(x_test), pos_label=1)
    f_score_macro += f_score
    if y_proba is not None:
        fp_rate, tp_rate, thresholds = roc_curve(y_test, y_proba[:, 1], pos_label=1)
        auc_score = auc(fp_rate, tp_rate)
    else:
        auc_score = 'not defined'
    logging.info('For label %s: precision %.2f, recall %.2f, f_score %.2f, auc %s' %
                 (labels_one[label_position], precision, recall, f_score, str(auc_score)))


    return f_score_macro / 2

init_logging()