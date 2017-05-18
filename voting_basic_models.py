from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, recall_score
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
import logging
import pickle
import fit_models
import itertools
import sys


def define_models(label):
    train_input, train_output, test_input, test_output = fit_models.fill_missed_modality('full_text_media_location')
    x_train, x_test = train_input, test_input
    y_train_0, y_test_0 = fit_models.get_groundtruth(train_output, test_output, 0)
    y_train_1, y_test_1 = fit_models.get_groundtruth(train_output, test_output, 1)
    y_train_2, y_test_2 = fit_models.get_groundtruth(train_output, test_output, 2)
    y_train_3, y_test_3 = fit_models.get_groundtruth(train_output, test_output, 3)
    # define models
    # GBM
    if label == 0:
        logging.info('Fit models for label {%d}...' % label)
        E_I_GBM = GradientBoostingClassifier(learning_rate=0.01, n_estimators=120, max_depth=11,
                                             min_samples_split=150, min_samples_leaf=61, max_features=37)
        E_I_GBM.fit(x_train, y_train_0)
        joblib.dump(E_I_GBM, 'models/gbm_0.pkl', compress=1)
        pickle.dump(E_I_GBM.predict(x_test), open('predictions/gbm_%d.pkl' % label, 'wb'))

        E_I_LG = LogisticRegression(C=1.0, penalty='l2')
        E_I_LG.fit(x_train, y_train_0)
        joblib.dump(E_I_LG, 'models/lg_0.pkl', compress=1)
        pickle.dump(E_I_LG.predict(x_test), open('predictions/lg_%d.pkl' % label, 'wb'))

        NB = GaussianNB()
        NB.fit(x_train, y_train_0)
        joblib.dump(NB, 'models/nb_0.pkl', compress=1)
        pickle.dump(NB.predict(x_test), open('predictions/nb_%d.pkl' % label, 'wb'))

        return E_I_GBM, E_I_LG, NB

    elif label == 1:
        S_N_GBM = GradientBoostingClassifier(learning_rate=0.05, n_estimators=30, max_depth=12,
                                             min_samples_split=300, min_samples_leaf=1, max_features=18)
        S_N_GBM.fit(x_train, y_train_1)
        joblib.dump(S_N_GBM, 'models/gbm_1.pkl', compress=1)
        pickle.dump(S_N_GBM.predict(x_test), open('predictions/gbm_%d.pkl' % label, 'wb'))

        S_N_LG = LogisticRegression(C=0.10, penalty='l2')
        S_N_LG.fit(x_train, y_train_1)
        joblib.dump(S_N_LG, 'models/lg_1.pkl', compress=1)
        pickle.dump(S_N_LG.predict(x_test), open('predictions/lg_%d.pkl' % label, 'wb'))

        NB = GaussianNB()
        NB.fit(x_train, y_train_0)
        joblib.dump(NB, 'models/nb_1.pkl', compress=1)
        pickle.dump(NB.predict(x_test), open('predictions/nb_%d.pkl' % label, 'wb'))

        return S_N_GBM, S_N_LG, NB

    elif label == 2:
        T_F_GBM = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120, max_depth=6,
                                             min_samples_split=525, min_samples_leaf=81, max_features=70)
        T_F_GBM.fit(x_train, y_train_2)
        joblib.dump(T_F_GBM, 'models/gbm_2.pkl', compress=1)
        pickle.dump(T_F_GBM.predict(x_test), open('predictions/gbm_%d.pkl' % label, 'wb'))

        T_F_LG = LogisticRegression(C=1.0, penalty='l1')
        T_F_LG.fit(x_train, y_train_2)
        joblib.dump(T_F_LG, 'models/lg_2.pkl', compress=1)
        pickle.dump(T_F_LG.predict(x_test), open('predictions/lg_%d.pkl' % label, 'wb'))

        NB = GaussianNB()
        NB.fit(x_train, y_train_2)
        joblib.dump(NB, 'models/nb_2.pkl', compress=1)
        pickle.dump(NB.predict(x_test), open('predictions/nb_%d.pkl' % label, 'wb'))

        return T_F_GBM, T_F_LG, NB

    elif label == 3:
        J_P_GBM = GradientBoostingClassifier(learning_rate=0.01, n_estimators=120, max_depth=12,
                                             min_samples_split=50, min_samples_leaf=71, max_features=65)
        J_P_GBM.fit(x_train, y_train_3)
        joblib.dump(J_P_GBM, 'models/gbm_3.pkl', compress=1)
        pickle.dump(J_P_GBM.predict(x_test), open('predictions/gbm_%d.pkl' % label, 'wb'))

        J_P_LG = LogisticRegression(C=0.50, penalty='l2')
        J_P_LG.fit(x_train, y_train_3)
        joblib.dump(J_P_LG, 'models/lg_3.pkl', compress=1)
        pickle.dump(J_P_LG.predict(x_test), open('predictions/lg_%d.pkl' % label, 'wb'))

        NB = GaussianNB()
        NB.fit(x_train, y_train_3)
        joblib.dump(NB, 'models/nb_3.pkl', compress=1)
        pickle.dump(NB.predict(x_test), open('predictions/nb_%d.pkl' % label, 'wb'))

        return J_P_GBM, J_P_LG, NB


class Voter():
    def __init__(self, weigths):
        self.weigths = weigths

    @property
    def weigths(self):
        return self.__weigths

    @weigths.setter
    def weigths(self, weigths):
        self.__weigths = weigths

    def predict(self, predictions_lists, size, predict_mode='soft'):
        result = list()
        for i in range(0, size):
            labels = [0, 0]

            for predictions_list, classif_index in zip(predictions_lists, range(0, len(predictions_lists))):
                if predict_mode == 'soft':
                    labels[predictions_list[i]] += self.__weigths[classif_index]
                elif predict_mode == 'hard':
                    labels[predictions_list[i]] += 1
            if labels[0] > labels[1]:
                result.append(0)
            else:
                result.append(1)
        return result


def get_voting_results_for_basic_models():
    train_input, train_output, test_input, test_output = fit_models.fill_missed_modality('full_text_media_location')
    for label in range(0, 4):
        x_train, x_test = train_input, test_input
        y_train, y_test = fit_models.get_groundtruth(train_output, test_output, label)
        f_macro = 0
        gbm, lg, nb = define_models(label)

        for gbm_prob in np.arange(0.1, 1.1, 0.2):
            for lg_prob in np.arange(0.1, 1.1, 0.2):
                for nb_prob in np.arange(0.1, 1.1, 0.2):
                    v_clf = VotingClassifier(estimators=[('gbm', gbm), ('lg', lg), ('nb', nb)], voting='soft',
                                             weights=[gbm_prob, lg_prob, nb_prob])
                    logging.info('Begin to fit voting for label {%d}...' % label)
                    logging.info('Weigths are %s' % str([gbm_prob, lg_prob, nb_prob]))
                    v_clf.fit(x_train, y_train)
                    f_measure = fit_models.log_results(x_test, y_test, v_clf.predict_proba(x_test), v_clf, label)
                    if f_measure > f_macro:
                        f_macro = f_measure
                        logging.info('NEW MAX FOUND %.3f' % f_macro)


def define_weigths(n_models):
    weigths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    weigths_list = list(itertools.product(weigths, repeat=n_models))
    return weigths_list


def log_results(y_test, y_pred, label_position, cur_max):
    # zero label
    precision_0 = precision_score(y_test, y_pred, pos_label=0)
    recall_0 = recall_score(y_test, y_pred, pos_label=0)
    f_score_0 = f1_score(y_test, y_pred, pos_label=0)
    f_score_macro = f_score_0

    # one label
    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    f_score_1 = f1_score(y_test, y_pred, pos_label=1)
    f_score_macro += f_score_1
    f_score_macro = f_score_macro / 2

    if f_score_macro > cur_max:
        logging.info('For label %s: precision %.2f, recall %.2f, f_score %.2f' %
                     (fit_models.labels_zero[label_position], precision_0, recall_0, f_score_0))
        logging.info('For label %s: precision %.2f, recall %.2f, f_score %.2f' %
                    (fit_models.labels_one[label_position], precision_1, recall_1, f_score_1))
    return f_score_macro


def load_results(models):
    results = list()
    for label in range(0, 4):
        results.append(list())

    for label in range(0, 4):
        for model in models:
            results[label].append(pickle.load(open('predictions/%s_%d.pkl' % (model, label), 'rb')))

    return results


def check_all_zero_weigths(weigths):
    for weigth in weigths:
        if weigth > 0:
            return False
    return True


def get_voting_results_for_all_models(label):
    train_input, train_output, test_input, test_output = fit_models.fill_missed_modality('full_text_media_location')

    models = ['512_2', '512_3', '512_4', '512_5', '512_6', '512_7', '512_8']
    weigths_list = define_weigths(len(models))
    results = load_results(models)
    voter = Voter(None)
    max_f_macro = 0
    y_train, y_test = fit_models.get_groundtruth(train_output, test_output, label)
    for weigths in weigths_list:
        if check_all_zero_weigths(weigths):
            continue
        voter.__setattr__('weigths', weigths)
        y_pred = voter.predict(results[label], len(y_test), 'soft')
        f_macro = log_results(y_test, y_pred, label, max_f_macro)
        if f_macro > max_f_macro:
            max_f_macro = f_macro
            logging.info('New max detected: %.3f' % f_macro)
            logging.info('Weigths: %s' % str(weigths))


def main(args):
    get_voting_results_for_all_models(int(args[0]))


if __name__ == "__main__":
    main(sys.argv[1:])