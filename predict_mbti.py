import fit_models
import pickle
from sklearn.metrics import accuracy_score

train_input, train_output, test_input, test_output = fit_models.fill_missed_modality('full_text_media_location')
models_for_label = ['512_7', '512_7', '512_7', '512_6']

pred_output = list()
for i in range(0, len(test_output)):
    pred_output.append('')

for label in range(0, 4):
    label_pred = pickle.load(open('predictions/%s_%d.pkl' % (models_for_label[label], label), 'rb'))
    for i in range(0, len(label_pred)):
        try:
            if label_pred[i] == 0:
                pred_output[i] += fit_models.labels_zero[label]
            else:
                pred_output[i] += fit_models.labels_one[label]
        except Exception:
            print ('label: %d' % label)
            print ('i: %d' % i)

accuracy = accuracy_score(test_output, pred_output)
print ('Accuracy: %.3f' % accuracy)
