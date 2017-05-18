from pymongo import MongoClient

# Mongo params
MONGO_PORT = 27017
MONGO_HOST = 'localhost'


def connect_to_database(host, port, db_name):
    # connects to mongo
    client = MongoClient(host, port)
    return client[db_name]

db = connect_to_database('localhost', MONGO_PORT, 'local')
users = db['full_text_media_location'].find()

user_ids = list()
for user in users:
    user_ids.append(user['_id'])

db = connect_to_database('localhost', MONGO_PORT, 'mbti_research')
feature_sets = ['text', 'liwc', 'lda', 'media', 'location']
periods = 10
for period in range(1, periods + 1):
    for feature in feature_sets:
        not_int_list = 0
        period_users = db['MBTI_%d_%s' % (period, feature)].find()
        for user in period_users:
            if user['_id'] not in user_ids:
                db['MBTI_%d_%s' % (period, feature)].remove({'_id': user['_id']})
                not_int_list += 1
        print ('period %d feature set %s found %d not in list users' % (period, feature, not_int_list))

all_users = db['users'].find()
count = 0
for user in all_users:
    if 'twitterUserName' not in user or user['twitterUserName'] not in user_ids:
        db['users'].remove({'_id': user['_id']})
        count += 1

print ('Remove %d users, left %d users' % (count, len(user_ids)))


