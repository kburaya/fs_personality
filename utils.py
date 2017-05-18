import uuid
import logging
from pymongo import MongoClient
import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import NMF

# Mongo params
MONGO_PORT = 27017
MONGO_HOST = 'localhost'
MONGO_DB = 'mbti_research'

labels_zero = ['E', 'N', 'F', 'P']
labels_one = ['I', 'S', 'T', 'J']

features_modalitites = {
    'text': 53,
    'liwc': 64,
    'lda': 50,
    'media': 1000,
    'location': 886
}


def init_logging(filename = None):
    if filename is None:
        filename = 'logs/%s.log' % str(uuid.uuid4())
    else:
        filename = 'logs/' + filename + '.log'
    print ('logs located in logs/%s' % filename)
    logging.basicConfig(filename=filename, filemode='w+', level=logging.DEBUG,
                        format='%(asctime)s %(message)s')


def load_resource(filename):
    filename = 'resources/%s.pkl' % filename
    return pickle.load(open(filename, 'rb'))


def save_resource(filename, object):
    filename = 'resources/%s.pkl' % filename
    return pickle.dump(object, open(filename, 'wb'))


def connect_to_database(host, port, db_name):
    client = MongoClient(host, port)
    return client[db_name]


def read_data():
    #  merge the users if they have id's in needed datasets
    global users, text, media, location, text_media, text_location, media_location, text_media_location
    users = pd.read_csv('data/users.csv', encoding="ISO-8859-1")
    text = pd.read_csv('data/text.csv', encoding="ISO-8859-1")
    media = pd.read_csv('data/media.csv', encoding="ISO-8859-1")
    location = pd.read_csv('data/location.csv', encoding="ISO-8859-1")

    text_location = location.merge(text, on='_id')
    text_location = text_location.merge(users, on='_id')
    text_location.to_csv('data/text_location.csv', encoding="ISO-8859-1")
    logging.info("Number of Twitter and Fourquare users: %d" % len(text_location))

    text_media = media.merge(text, on='_id')
    text_media = text_media.merge(users)
    text_media.to_csv('data/text_media.csv', encoding="ISO-8859-1")
    logging.info("Number of Twitter and Instagram users: %d" % len(text_media))

    media_location = location.merge(media, on='_id')
    media_location = media_location.merge(users, on='_id')
    media_location.to_csv('data/media_location.csv', encoding="ISO-8859-1")
    logging.info("Number of Instagram and Foursquare users: %d" % len(media_location))

    text_media_location = location.merge(media, on='_id')
    text_media_location = text_media_location.merge(text, on='_id')
    text_media_location = text_media_location.merge(users, on='_id')
    text_media_location.to_csv('data/text_media_location.csv', encoding="ISO-8859-1")
    logging.info("Number of Twitter, Instagram and Foursquare users: %d" % len(text_media_location))

    text = text.merge(users, on='_id')
    logging.info("number of Twitter users: %d" % len(text))

    media = media.merge(users, on='_id')
    logging.info("number of Instagram users: %d" % len(media))

    location = location.merge(users, on='_id')
    logging.info("number of Foursquare users: %d" % len(location))


def read_and_fill_missed_data():
    # merge the users and fill 0 for missing data
    # return the list of users
    global users, text, media, location, text_media, text_location, media_location, text_media_location
    users = pd.read_csv('data/users.csv', encoding="ISO-8859-1")
    text = pd.read_csv('data/all_periods_text.csv', encoding="ISO-8859-1")
    liwc = pd.read_csv('data/all_periods_liwc.csv', encoding="ISO-8859-1")
    lda = pd.read_csv('data/all_periods_lda.csv', encoding="ISO-8859-1")
    media = pd.read_csv('data/all_periods_media.csv', encoding="ISO-8859-1")
    location = pd.read_csv('data/location.csv', encoding="ISO-8859-1")

    text_media_location = text.merge(liwc, how='left', on='_id')
    text_media_location = text_media_location.merge(lda, how='left', on='_id')
    text_media_location = text_media_location.merge(media, how='left', on='_id')
    text_media_location = text_media_location.merge(location, how='left', on='_id')
    text_media_location.fillna(0, inplace=True)
    text_media_location.to_csv('data/full_text_media_location.csv')

    users = list(text_media_location['_id'])
    return


def download_data():
    try:
        train_input = pickle.load(open('resources/train_input.pkl', 'rb'))
        train_output = pickle.load(open('resources/train_output.pkl', 'rb'))
        test_input = pickle.load(open('resources/test_input.pkl', 'rb'))
        test_output = pickle.load(open('resources/test_output.pkl', 'rb'))
        return train_input, train_output, test_input, test_output
    except:
        raise FileNotFoundError('There is no prepared window data!')


def fill_missed_modality(users_list, features_set):
    # get the collection of full modalities for all users
    # return train/test sets with NMF
    # try:
    #     return download_data()
    # except FileNotFoundError:
    #     logging.info('There is no prepared data on disk!')
    db = connect_to_database(MONGO_HOST, MONGO_PORT, MONGO_DB)
    try:
        transformed_data = pickle.load(open('resources/transformed_%s.pkl' % features_set, 'rb'))
        return transformed_data
    except FileNotFoundError:
        logging.info('There is no prepared NMF data on disk!')
        null_users = 0
        users_features = list()
        features_order = list()  # to save the order of features in NMF transformed data for later operations
        for user in users_list:
            features = db['all_%s' % features_set].find_one({'_id': user})
            if features is None:
                users_features.append(np.array([0] * features_modalitites[features_set]))
                null_users += 1
                continue
            feature_vector = list()
            if len(features_order) == 0:
                features_order = list(features.keys())
                features_order.remove('_id')

            for feature in features_order:
                if features[feature] < 0:  # FIXME it happens only for sentiment score here
                    feature_vector.append((-1.0) * features[feature])
                else:
                    feature_vector.append(features[feature])
            users_features.append(np.array(feature_vector))

        logging.info('Found %d users with %d features' % (len(users_list) - null_users, len(users_features[0])))
        logging.info('Begin to do NMF for %s' % features_set)
        R = np.array(users_features)
        model = NMF(init='random', random_state=0)
        W = model.fit_transform(R)
        H = model.components_
        transformed_data = np.dot(W, H)

        logging.info('NMF completed')
        pickle.dump(transformed_data, open('resources/transformed_%s.pkl' % features_set, 'wb'))
        pickle.dump(features_order, open('resources/features_order_%s.pkl' % features_set, 'wb'))

    # train_input, train_output, test_input, test_output = list(), list(), list(), list()
    # for user_pos in range(0, len(transformed_data)):
    #     user_groundtruth = db['users'].find_one({'twitterUserName': users_list[user_pos]})
    #     if user_groundtruth is None:
    #         logging.error('Found None user {%s}' % users_list[user_pos])
    #         continue
    #     if user_groundtruth['set'] == 'train':
    #         train_input.append(transformed_data[user_pos])
    #         train_output.append(user_groundtruth['mbti'])
    #     else:
    #         test_input.append(transformed_data[user_pos])
    #         test_output.append(user_groundtruth['mbti'])
    #
    # logging.info('Saving data...')
    # pickle.dump(train_input, open('resources/train_input.pkl', 'wb'))
    # pickle.dump(train_output, open('resources/train_output.pkl', 'wb'))
    # pickle.dump(test_input, open('resources/test_input.pkl', 'wb'))
    # pickle.dump(test_output, open('resources/test_output.pkl', 'wb'))
    # logging.info('Data saved. Found {%d} train, {%d} test users' % (len(train_output), len(test_output)))
    # return np.array(train_input), train_output, np.array(test_input), test_output


def average_periods(periods_num, features_set):
    db = connect_to_database('localhost', MONGO_PORT, 'mbti_research')
    average_features = dict()
    periods_per_user = dict()

    for p in range(0, periods_num):
        # iterate through periods and collect the sum of features
        period_features = db['MBTI_%d_%s' % (p + 1, features_set)].find()
        for user in period_features:
            if user['_id'] not in average_features:
                average_features[user['_id']] = dict()
            for feature in user:
                if feature != '_id':
                    if feature not in average_features[user['_id']]:
                        average_features[user['_id']][feature] = user[feature]
                    else:
                        average_features[user['_id']][feature] += user[feature]
            if user['_id'] not in periods_per_user:
                periods_per_user[user['_id']] = 1
            else:
                periods_per_user[user['_id']] += 1

    average_features_collection = 'all_%s' % features_set
    print('Collect {%d} users' % len(periods_per_user))
    print('Save into collection {%s}' % average_features_collection)
    for user in periods_per_user:
        user_average = dict()
        for feature in average_features[user]:
            user_average[feature] = average_features[user][feature] / periods_per_user[user]
        user_average['_id'] = user
        db[average_features_collection].insert(user_average)
