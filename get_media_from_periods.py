from pymongo import MongoClient

# Mongo params
MONGO_PORT = 27017
MONGO_HOST = '172.29.30.160'
MONGO_DB = 'user-profiling'

def connect_to_database(host, port, db_name):
    # connects to mongo
    client = MongoClient(host, port)
    return client[db_name]

db = connect_to_database(MONGO_HOST, MONGO_PORT, 'part_mbti_text_features')
periods = 10
features_dimension = 64

user_periods_number = dict()  # user -> in how many periods he was
media_features = dict()

for period in range(1, periods + 1):
    period_users = db['tweetsMBTI_%d_liwc' % period].find()
    print ('process period %d' % period)
    for period_user in period_users:
        if period_user['_id'] not in media_features:
            media_features[period_user['_id']] = [0] * features_dimension
            user_periods_number[period_user['_id']] = 0

        for pos, feature in zip(range(0, features_dimension), period_user):
            if feature != '_id':
                media_features[period_user['_id']][pos] += period_user[feature]
        user_periods_number[period_user['_id']] += 1

for user in media_features:
    features_to_db = dict()
    for i in range(0, features_dimension):
        features_to_db['liwc_' + str(i)] = media_features[user][i] / user_periods_number[user]
        # features_to_db['location_' + str(i)] = media_features[user][i]
    features_to_db['_id'] = user
    db['all_periods_liwc'].insert(features_to_db)
