
# coding: utf-8

import numpy as np
import pandas as pd
import tflearn
import os
from tqdm import tqdm
from random import shuffle
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression
from sklearn.preprocessing import StandardScaler


scaler=StandardScaler()
TRAIN_DIR ='train.csv'
TEST_DIR='test.csv'
lr = 1e-3
model_name = 'forest.model'
df_train = pd.read_csv(TRAIN_DIR, index_col='Id')
df_test = pd.read_csv(TEST_DIR, index_col='Id')
result = ["1 - Spruce/Fir",
"2 - Lodgepole Pine",
"3 - Ponderosa Pine",
"4 - Cottonwood/Willow",
"5 - Aspen",
"6 - Douglas-fir",
"7 - Krummholz"]
type_list = [0,0,0,0,0,0,0]

# print(df.head())

def prepare_training_data(filename):
    labels = []
    features = []
    soiltype=[]
    f=1
    file = open(filename, "r")
    for line in file:
        row = line.split(',')
        if f is 1:
            f=0
            print(len(row))
            continue
        else:
            a = np.zeros(7)
            a[int(row[55])-1]=1
            labels.append(a)
            soil=list(map(float,row[15:55]))
            soiltype.append(soil.index(1)+1)
#             soiltype = soil.index(1.0)
            features.append([float(x) for x in row[1:15]])
            features.append(float(soil.index(1)+1))
    file.close()
    return np.array(labels, dtype=np.int64), np.array(features, dtype=np.float64), np.array(soiltype,dtype=np.float64)


def prepare_test_data(filename):
    features = []
    f=1
    file = open(filename, "r")
    for line in file:
        row = line.split(',')
        if f is 1:
            f=0
            continue
        else:
            features.append([float(x) for x in row[1:55]])
    file.close()
    return np.array(features, dtype=np.float64)


train_lab, train_feat, soilt = prepare_training_data(TRAIN_DIR)
print(train_lab[:5])
print(train_lab.shape)
print(soilt[:1])
print(soilt.shape)

# train_lab-=1
# labels_1 = np.zeros((len(train_lab),7))
# labels_1[np.arange(len(train_lab)), train_lab] = 1.0

# print(features_1[:5])
# print(train_lab[:20])
print(train_feat[:2])
print(train_feat.shape)


#NETWORK MODEL
network = input_data(shape=[None,54], name='input')

network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.8)

network = fully_connected(network, 128, activation='relu')
network = dropout(network, 0.8)

# network = fully_connected(network, 64, activation='relu')
# network = dropout(network, 0.8)

# network = fully_connected(network, 32, activation='relu')
# network = dropout(network, 0.8)

network = fully_connected(network, 7, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(network, tensorboard_dir='log')


model.fit({'input': train_feat}, {'targets': train_lab}, n_epoch=20, snapshot_step=500, show_metric=True, run_id=model_name)


test_data = prepare_test_data(TEST_DIR)
print("Test Data Prepared!")


model_out = model.predict(test_data)
print("Beginning Predictions!")
for line in model_out:
    type_no=np.argmax(line)
    type_list[int(type_no)-1]+=1
#     print(result[int(type_no)])
print(type_list)


# model.save(model_name)
# print("Model Saved!")


# with open("submission_file.csv", "w") as f:
#     f.write("id,Cover_type\n")
# with open("submission_file.csv", "a") as f:
#     model_out = model.predict(test_data)
# #     print("Beginning Predictions!")
#     num=15121
#     for line in model_out:
#         type_no=np.argmax(line)
#         type_list[int(type_no)-1]+=1
#         f.write("{},{}\n".format(num,type_no+1))
#         num+=1
#     #     print(result[int(type_no)])
# #     print(type_list)

