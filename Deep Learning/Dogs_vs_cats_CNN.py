import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import os
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import time

DIR='C:/Users/Asus/Downloads/ML DATASETS/Dogs Vs Cats/'
CATEGORIES = ['dog', 'cat']
train_path=os.path.join(DIR, 'train')
IMG_SIZE = 100

for img in tqdm(os.listdir(train_path)):
    label = img.split('.')[0]
    img_array = cv2.imread(os.path.join(train_path, img), cv2.IMREAD_GRAYSCALE)
    plt.imshow(img_array, cmap='red')
    plt.show()
    break

print(img_array)

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), 1)
plt.imshow(new_array, cmap="red")
plt.show()

training_data = []

def create_train_data():
    for img in tqdm(os.listdir(train_path)):
        try:
            label = img.split('.')[0]
            class_num = CATEGORIES.index(label)
            img_array = cv2.imread(os.path.join(train_path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass
        
create_train_data()
print(len(training_data))
random.shuffle(training_data)

X=[]
y=[]
for features,labels in training_data:
    X.append(features)
    y.append(labels)
    
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

NAME = "dogs_vs_cats-CNN-64x2-{}"
tensorboard = TensorBoard(log_dir='/logs/{}'.format(NAME))
X = X/255.0

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
             optimizer='adam',
             metrics=['accuracy'])

model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])

predict=['Dog', 'Cat']
def prepare(filepath):
    image_array=cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_img = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    return new_img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
prediction=model.predict([prepare('D:\250px-Gatto_europeo4.jpg')])
print(predict[int(prediction[0][0])])
   
