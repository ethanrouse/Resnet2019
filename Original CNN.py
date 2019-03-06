
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
from scipy import misc
from natsort import natsorted
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data_dir = 'X:/SchizophreniaCNN'
labels_df = pd.read_csv(data_dir + '/Labels.csv', index_col = 0, encoding = "ISO-8859-1")
labels_df.head() 


# In[3]:


label_list = labels_df['Class'].tolist()
for label in label_list[:5]:
    print(label)
print(len(label_list))
print(label_list[:10])


# In[4]:


# Create one hot encoding for labels
for i in range(len(label_list)):
    if label_list[i] == 1:
        label_list[i] = np.array([0,1])
    elif label_list[i] == 0:
        label_list[i] = np.array([1,0])
print(label_list[:5])


# In[5]:


patients = os.listdir(data_dir + '/PNG Nifti Files')
print(patients[:10])
print(label_list[:10])
print(len(patients))


# In[6]:


import math

image_pixel_dim =64
slice_count = 77

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def mean(l):
    return sum(l)/len(l)


# In[2]:


def process_data(patient, labels_df, px_dim = 64, slices = 77):
    label = labels_df.get_value(patient, 'Class')
    path = data_dir + '/PNG Nifti Files/' + patient    
    
    slice_paths = []
    slices = []
    for s in os.listdir(path):
        slice_paths.append(path + '/' + s)
    slice_paths = natsorted(slice_paths)
    slice_paths = slice_paths[47:201]
    
    for num, spath in enumerate(slice_paths):
        slice_read = misc.imread(spath)
        slices.append(slice_read)
    #print(len(slices))
    new_slices = []
    # Creates the resized slices
    slices = [cv2.resize(image_pixel_dim,image_pixel_dim,np.array(each_slice)) for each_slice in slices]
    # Simply divide the list up into equal parts based on the slice_count value
    chunk_sizes = math.ceil(len(slices) / slice_count) 
    
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)
    #print(len(new_slices))
    #print(np.asarray(new_slices).shape)    

    print('Patient: ', patient)
    print('Label: ', label)
    
    if label == 1:
        label = np.array([0,1])
    elif label == 0:
        label = np.array([1,0])
    #fig = plt.figure()
    #for num, each_slice in enumerate(new_slices):
    #    y = fig.add_subplot(8,8,num+1)
    #    y.imshow(each_slice)
    #plt.show()    
    return np.array(new_slices), label


# In[3]:


data_array = []

for num, patient in enumerate(patients):
    if num%10 == 0:
        print(num)
        
    img_data, label = process_data(patient, labels_df, px_dim = image_pixel_dim, slices = slice_count)
    data_array.append([img_data, label])
    
    np.save('patientdata--{}--{}--{}.npy'.format(image_pixel_dim, image_pixel_dim, slice_count), data_array)


# In[8]:


# CNN Setup
n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.6


# In[9]:


# Conv3D and Maxpool3d layer setup
def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window as you slide about
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')


# In[10]:


# Final CNN structure

image_pixel_dim = 64
slice_count = 22

def convolutional_neural_network(x):
    #          # 4 x 4 x 4 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([4,4,4,1,32])),
               #4 x 4 x 4 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([4,4,4,32,64])),
               'W_conv3':tf.Variable(tf.random_normal([4,4,4,64,128])),
               'W_conv4':tf.Variable(tf.random_normal([4,4,4,128,256])),
               'W_conv5':tf.Variable(tf.random_normal([4,4,4,256,512])),
               'W_fc':tf.Variable(tf.random_normal([2*2*1*512,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_conv3':tf.Variable(tf.random_normal([128])),
              'b_conv4':tf.Variable(tf.random_normal([256])),
              'b_conv5':tf.Variable(tf.random_normal([512])),
              'b_fc':tf.Variable(tf.random_normal([1024])),
              'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X             image Y        image Z
    x = tf.reshape(x, shape=[-1, image_pixel_dim, image_pixel_dim, slice_count, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)

    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)
    
    conv3 = tf.nn.relu(conv3d(conv2, weights['W_conv3']) + biases['b_conv3'])
    conv3 = maxpool3d(conv3)
    
    conv4 = tf.nn.relu(conv3d(conv3, weights['W_conv4']) + biases['b_conv4'])
    conv4 = maxpool3d(conv4)
    
    conv5 = tf.nn.relu(conv3d(conv4, weights['W_conv5']) + biases['b_conv5'])
    conv5 = maxpool3d(conv5)
    
    fc = tf.reshape(conv5,[-1, 2*2*1*512])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


# In[1]:


data_array = np.load('patientdata--64--64--22.npy')

train_data = data_array[:-30]
validation_data = data_array[-30:]
accuracy_array = []

def train_neural_network(x):
    count = 0
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels=y))
    learn_rate = tf.train.exponential_decay(start_learn_rate, global_step, 100000, 0.5, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
    
    hm_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        successful_runs = 0
        total_runs = 0
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    pass
                    #print(str(e))
            
            if epoch % 10 == 0:
                print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
            acc = accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]})
            count = count + acc
            accuracy_array.append(acc)
            count_invert = count + (1 - acc)
            
        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        print('Accuracy on Train Data: ',accuracy.eval({x:[i[0] for i in train_data], y:[i[1] for i in train_data]}))
        print('Avg Accuracy: ', count/hm_epochs)
        print('Inverted Avg Accuracy (acc - 1): ', count_invert/hm_epochs)
        print('Standard Deviation: ', np.std(accuracy_array))

# Run this locally:
# train_neural_network(x)


# In[14]:


train_neural_network(x)

