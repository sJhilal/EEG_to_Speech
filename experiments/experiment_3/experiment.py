from tensorflow.keras.models import load_model
import json
import os
import tensorflow as tf
import numpy as np
from custom_code.data.dataset_builder import TFRecordsDatasetBuilder, Default2EnvBatchEqualizer, TFRecordsDatasetBuilder_2, Default2EnvBatchEqualizer_2
from experiments.experiment_3.models import acc_wrapper, categorical_balanced


weights_train = np.array([1 / 0.24766, 1/0.7523])
weights_test = np.array([1/0.24984, 1/0.75015])

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_folder = root + "/dataset/LSTM_data"
results_path = root + "/experiments/experiment_3/output"
model_path = root + "/experiments/experiment_3/model_classifier"

model = load_model(results_path+"/model_classifier.h5", custom_objects={'loss': categorical_balanced(weights_train), 'weighted_acc':[tf.keras.metrics.CategoricalAccuracy(), acc_wrapper(weights_test)]})


os.makedirs(results_path, exist_ok=True)

ds_creator = TFRecordsDatasetBuilder(folder=data_folder)
ds_creator_2 = TFRecordsDatasetBuilder_2(folder=data_folder)

window_length = 320   # this is equal to 5 seconds decision window
# only data of one subject "B30K04" is used in this script. remove filters line in the below dataset creation
# functions to use all the subjects available
# batch size 1 allows us to easily seperate between classes in prediction
one_hot_ds = ds_creator_2.prepare(
    "test",
    batch_size=1,
    window=window_length,
    window_overlap=0.9,
    batch_equalizer=Default2EnvBatchEqualizer_2(),
)

miss_match_ds = ds_creator.prepare(
    "test",
    batch_size=1,
    window=window_length,
    window_overlap=0.9,
    batch_equalizer=Default2EnvBatchEqualizer(),
)


acc_per_sub={}
misses=0
matches=0
for subject, ds_test in one_hot_ds.items():
    sub = subject
    ds = ds_test
    x = model.predict(ds_test)
    sub_preds = np.zeros(len(x),int)
    new_x = np.zeros((len(x),320,1))
    for z in range(len(x)):
        for e in range(320):
            if x[z][e][1] >= x[z][e][0]:
                new_x[z][e]=1
    X_train = list(map(lambda x: x[0], miss_match_ds[subject]))
    for u in range(len(x)):
        good_hits = 0
        bad_hits = 0
        good = X_train[u][1][0].numpy().flatten()
        bad = X_train[u][1][1].numpy().flatten()
        pred_anypho = new_x[u].flatten()
        for j in range(320):
            if pred_anypho[j]==good[j]:
                good_hits+=1
            if pred_anypho[j]==bad[j]:
                bad_hits+=1
        if good_hits >= bad_hits:
            sub_preds[u]=1
            matches+=1
        else:
            misses+=1
    acc_per_sub[subject]=np.count_nonzero(sub_preds)/len(sub_preds)


total=matches+misses
overall_acc=matches/total
print("Overall accuracy is "+str(overall_acc))

with open(results_path + "/acc_per_sub.json", "w") as fp:
    json.dump(acc_per_sub, fp)


"""
Code below picks the match segment based on highest correlation instead of most correct points

"""


# acc_per_sub={}
# misses=0
# matches=0
# for subject, ds_test in one_hot_ds.items():
#     sub = subject
#     ds = ds_test
#     x = model.predict(ds_test)
#     sub_preds = np.zeros(len(x),int)
#     new_x = np.zeros((len(x),320,1))
#     for z in range(len(x)):
#         for e in range(320):
#             if x[z][e][1] >= x[z][e][0]:
#                 new_x[z][e]=1
#     X_train = list(map(lambda x: x[0], miss_match_ds[subject]))
#     for u in range(len(x)):
#         good = X_train[u][1][0].numpy().flatten()
#         bad = X_train[u][1][1].numpy().flatten()
#         pred_anypho = new_x[u].flatten()
#         good_corr = np.corrcoef(pred_anypho,good)[0][1]
#         bad_corr = np.corrcoef(pred_anypho,bad)[0][1]
#         if good_corr >= bad_corr:
#             sub_preds[u]=1
#             matches+=1
#         else:
#             misses+=1
#     acc_per_sub[subject]=np.count_nonzero(sub_preds)/len(sub_preds)
