
# coding: utf-8

# In[1]:


import numpy as np
from keras.layers import Dense,Dropout,LSTM,CuDNNLSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import Adam


# In[2]:


filename="Harrytest.txt"
raw_text = open("D:/TextRNN/{}".format(filename)).read()
raw_text = raw_text.lower()


# In[3]:


chars = sorted(list(set(raw_text)))
chars_to_int = dict((c,i) for i,c in enumerate(chars))


# In[4]:


print(chars)


# In[5]:


n_chars = len(raw_text)
n_vocab = len(chars)

print("Total Characters: {}".format(n_chars))
print("Total Vocab: {}".format(n_vocab))


# In[6]:


seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i : i+seq_length]
    seq_out = raw_text[i+seq_length]
    dataX.append([chars_to_int[char] for char in seq_in])
    dataY.append(chars_to_int[seq_out])
    
n_patterns = len(dataX)
print ("Total Patterns: {}".format(n_patterns))


# In[7]:


print(dataX[:2])
print(dataY[:10])


# In[8]:


X = np.reshape(dataX, (n_patterns, seq_length, 1)) #reshape
X = X/float(n_vocab)
y = np_utils.to_categorical(dataY)


# In[9]:


print(X[:2])
print(y[:10])


# In[10]:


model = Sequential()
model.add(CuDNNLSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

opt = Adam(lr=1e-3, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt)


# In[11]:


filepath = "D:/TextRNN/Weights/TextRNN_wgt_improv-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# In[12]:


# model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)


# In[13]:


#GENERATING TEXT NOW BABYY!!


# In[20]:


filename = filepath = "D:/TextRNN/Weights/TextRNN_wgt_improv-35-1.4144.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer=opt)


# In[21]:


int_to_char = dict((i,c) for i,c in enumerate(chars))


# In[22]:


import sys

start = np.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print("SEED:\n")
print("\"",''.join([int_to_char[value] for value in pattern]), "\"")

predict_len = 1000

print("\nPREDICTIONS:\n")
for i in range(predict_len):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x/float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(result, end='')
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    
print("\n DONE!")

