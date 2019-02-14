
# coding: utf-8

# In[2]:


import keras
import datetime
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import numpy as np
from resnet3d import Resnet3DBuilder


# In[88]:


# Set network parameters for later use
batch_size = 5
nb_classes = 2
epochs = 50

img_rows, img_cols, slices = 64, 64, 64


# # Some preprocessing to fit the resnet model

# In[89]:


# Load from pre created data file
data_array = np.load('X:\Directed Study\ResNet\patientdata--64--64--64.npy')

# Seperate data into training and validation data
train_data = data_array[:-40]
validation_data = data_array[-40:]

X_train = train_data[0:,0]

Y_train = []

# Get the one hot encoded labels from the training data
for i in range(len(train_data)):
    Y_train.append(train_data[i,1])
    
X_train_formatted = []
    
# Get all image data from the training set
for i in range(len(X_train)):
    X_train_formatted.append(X_train[i])
    
# Convert to numpy array for ease of use
Y_train_final = np.asarray(Y_train)
X_train_final = np.asarray(X_train_formatted)

# Reshape the 4D input into 5D for input to the network
X_train_final = X_train_final.reshape((X_train_final.shape[0],64,64,64,1))
print("X Train: " + str(X_train_final.shape))
print("Y Train: " + str(Y_train_final.shape))


# In[90]:


# Doing the same as above just with the validation data
Y_test = []

for i in range(len(validation_data)):
    Y_test.append(validation_data[i,1])

X_test = validation_data[0:,0]
X_test_formatted = []

for i in range(len(X_test)):
    X_test_formatted.append(X_test[i])
    
Y_test_final = np.asarray(Y_test)
X_test_final = np.asarray(X_test_formatted)
X_test_final = X_test_final.reshape((X_test_final.shape[0],64,64,64,1))
print("X Test: " + str(X_test_final.shape))
print("Y Test: " + str(Y_test_final.shape))


# In[91]:


lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10, min_lr=1e-6)

early_stopper = EarlyStopping(monitor='val_loss',
                              min_delta=0.001,
                              patience=50)

#csv_logger = CSVLogger('output/{}_{}.csv'.format(datetime.datetime.now().isoformat(), "Resnet2019"))


# # Model Compilation and Training

# In[92]:


# Set reg factor and use the resnet builder to build an architecture
regularization_factor = 2.5e-2
model = Resnet3DBuilder.build_resnet_50((img_rows,img_cols,slices,1),2,regularization_factor)

# Compile the model
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])


# In[93]:


model.fit(X_train_final, Y_train_final,
         epochs=epochs,
         validation_data=(X_test_final, Y_test_final),
         shuffle=True,
         callbacks=[lr_reducer, early_stopper])


# In[94]:


model.summary()

