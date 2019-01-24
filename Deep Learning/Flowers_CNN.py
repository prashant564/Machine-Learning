
# coding: utf-8

# In[1]:


import tensorflow
import numpy as np
import cv2
import tflearn
import os
from random import shuffle
from tqdm import tqdm
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

train_dir = 'C:/Users/Asus/Downloads/ML DATASETS/Flowers/flowers/'
test_dir = ''
ht = 150
wid = 150
lr = 1e-3
names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
Model_name = "flowers_CNN"
# label_onehot = [0,0,0,0,0]


# In[2]:


def label_image(path):
    label_onehot = [0,0,0,0,0]
    if path is 'daisy':
        label_onehot[0]+=1
    elif path is 'dandelion':
        label_onehot[1]+=1
    elif path is 'rose':
        label_onehot[2]+=1
    elif path is 'sunflower':
        label_onehot[3]+=1
    elif path is 'tulip':
        label_onehot[4]+=1
    return label_onehot


# In[3]:


def create_train_data():
    train_data = []
    for ftype in names:
        path = os.path.join(train_dir, ftype)
        for img in tqdm(os.listdir(path)):
            type1=str(img.split('.')[1])
#             print(type1)
            path1 = os.path.join(path, img)
#             print(path1)
            label = label_image(ftype)
            img = cv2.resize(cv2.imread(path1, cv2.IMREAD_GRAYSCALE), (wid, ht))
#             print(img)
            train_data.append([np.array(img, dtype=np.float64), np.array(label, dtype=np.float64)])
    shuffle(train_data)
#     np.save('traindata.npy', train_data)
    return train_data
            
      


# In[4]:


def process_training_data():
   test_data = []
   for ftype in names:
       path1 = os.path.join(test_dir,ftype)  
       for img in os.listdir(path1):
           print('Type:- ',ftype)
           path1 = os.path.join(path1, img)
           label = label_image(ftype)
           img = cv2.resize(cv2.imread(path1, cv2.IMREAD_GRAYSCALE), (wid, ht))
           test_data.append([np.array(img), np.array(label)])
   shuffle(train_data)
#     np.save('traindata.npy', train_data)
   return test_data


# In[5]:


#NETWORK MODEL
convnet = input_data(shape=[None, wid, ht, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=lr, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


# In[6]:


train_data = create_train_data()
print(train_data[:2])
train = train_data[:-2000]
test = train_data[-2000:]
print("Data Created!")
print(train[:3])


# In[7]:


X = np.array([i[0] for i in train_data]).reshape(-1, wid, ht, 1)
Y = [i[1] for i in train_data]


# In[8]:


# test_x = np.array([i[0] for i in test]).reshape(-1, wid, ht, 1)
# test_y = [i[1] for i in test]


# In[9]:


model.fit({'input': X}, {'targets': Y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id=Model_name)

