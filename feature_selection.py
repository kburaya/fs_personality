import utils
import pandas as pd
from scipy.stats import pearsonr
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import logging

db = utils.connect_to_database(utils.MONGO_HOST, utils.MONGO_PORT, utils.MONGO_DB)
data = pd.read_csv('data/full_text_media_location.csv', encoding="ISO-8859-1")
users = db['users'].distinct('twitterUserName')
# feature_sets = ['text', 'liwc', 'lda', 'media', 'location']
feature_sets = ['text', 'liwc', 'lda']
fs_amount = {
    'text': 0.5,
    'liwc': 0.5,
    'lda': 0.5,
    'media': 0.05,
    'location': 0.05
}
periods = range(1, 11)


def get_groundtruth(label):
    try:
        return utils.load_resource('groundtruth_%s' % label)
    except FileNotFoundError:
        groundtruth = list()
        for user in users:
            user_object = db['users'].find_one({'twitterUserName': user})
            groundtruth.append(user_object['mbti'][label])
        utils.save_resource('groundtruth_%s' % label, groundtruth)
        return groundtruth


def print_features_name(features_mask, feature_set):
    features_order = utils.load_resource('features_order_%s' % feature_set)
    selected_features = list()
    for (feature, feature_mask) in zip(features_order, features_mask):
        if feature_mask:
            selected_features.append(feature)
    return selected_features


def do_correlation(features_set, label):
    utils.init_logging(filename='do_correlation_fun')
    groundtruth = get_groundtruth(label)
    features = utils.load_resource('transformed_%s' % features_set)
    data = pd.DataFrame(features)
    data['label'] = groundtruth
    data.corr().to_csv('resources/corr_%s.csv' % features_set)


def do_feature_selection_k_best(features_set, features_num, label):
    # utils.init_logging(filename='do_feature_selection_fun')
    groundtruth = get_groundtruth(label)
    try:
        selector = utils.load_resource('selector_%s_%d_%d' % (features_set, features_num, label))
    except FileNotFoundError:
        selector = SelectKBest(k=features_num, score_func=mutual_info_classif)
    features = utils.load_resource('transformed_%s' % features_set)
    selector.fit(features, groundtruth)
    logging.info('Select best %d feature for %s set, %d label' % (features_num, features_set, label))
    logging.info(print_features_name(selector.get_support(), features_set))
    utils.save_resource('selector_%s_%d_%d' % (features_set, features_num, label))
    return selector


def apply_and_save_fs():
    utils.init_logging(filename='apply_and_save_fs_fun')
    for label in range(0, 4):
        for feature_set in feature_sets:
            new_feature_dim = int(utils.features_modalitites[feature_set] * fs_amount[feature_set])
            selector = do_feature_selection_k_best(features_set=feature_set,
                                                   features_num=new_feature_dim,
                                                   label=label)
            for period in periods:
                apply_fs_for_period(selector=selector, period=period,
                                    feature_set=feature_set, label=label)

    return


def apply_fs_for_period(selector, period, feature_set, label):
    logging.info('Apply FS for %s set, %d period, %d label' % (feature_set, period, label))
    db_to_load = utils.connect_to_database('localhost', 27017, 'mbti_research')
    db_to_save = utils.connect_to_database('localhost', 27017, 'mbti_research_fs')

    old_features = db_to_load['MBTI_%d_%s' % (period, feature_set)].find()
    features_order = utils.load_resource('features_order_%s' % feature_set)
    for user in old_features:
        features_list = list()
        for feature in features_order:
            features_list.append(user[feature])
        new_features = selector.transform(features_list)
        new_features = list(new_features[0])

        user_object = dict()
        for (feature, index) in zip(new_features, range(0, len(new_features))):
            user_object['%s_%d' % (feature_set, index)] = feature
        user_object['_id'] = user['_id']

        db_to_save['fs_%s_%d_%d' % (feature_set, period, label)].insert(user_object)
    return


# for feature_set in feature_sets:
#     utils.fill_missed_modality(users_list=users, features_set=feature_set)
apply_and_save_fs()

