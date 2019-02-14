
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
import scipy.misc
from natsort import natsorted
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


data_dir = 'X:/SchizophreniaCNN'
labels_df = pd.read_csv(data_dir + '/Labels.csv', index_col = 0, encoding = "ISO-8859-1")
labels_df.head() 


# In[11]:


label_list = labels_df['Class'].tolist()
for label in label_list[:5]:
    print(label)
print(len(label_list))
print(label_list[:10])


# In[12]:


# Create one hot encoding for labels
for i in range(len(label_list)):
    if label_list[i] == 1:
        label_list[i] = np.array([0,1])
    elif label_list[i] == 0:
        label_list[i] = np.array([1,0])
print(label_list[:5])


# In[13]:


patients = os.listdir(data_dir + '/PNG Nifti Files')
print(patients[:10])
print(label_list[:10])
print(len(patients))


# In[14]:


import math

image_pixel_dim =64
slice_count = 64

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def mean(l):
    return sum(l)/len(l)


# In[15]:


def process_data(patient, labels_df, px_dim = 64, slices = 64):
    label = labels_df.get_value(patient, 'Class')
    path = data_dir + '/PNG Nifti Files/' + patient    
    
    slice_paths = []
    slices = []
    for s in os.listdir(path):
        slice_paths.append(path + '/' + s)
    slice_paths = natsorted(slice_paths)
    slice_paths = slice_paths[32:-32] #Used to omit some useless slices from the scans
    
    for num, spath in enumerate(slice_paths):
        slice_read = scipy.misc.imread(spath)
        slices.append(slice_read)
    #print(len(slices))
    new_slices = []
    # Creates the resized slices
    slices = [cv2.resize(np.array(each_slice),(image_pixel_dim,image_pixel_dim)) for each_slice in slices]
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


# # Processing all of the data

# In[17]:


data_array = []

for num, patient in enumerate(patients):
    if num%10 == 0:
        print(num)
        
    img_data, label = process_data(patient, labels_df, px_dim = image_pixel_dim, slices = slice_count)
    data_array.append([img_data, label])
    
    np.save('patientdata--{}--{}--{}.npy'.format(image_pixel_dim, image_pixel_dim, slice_count), data_array)


# In[ ]:


print(np.asarray(data_array.shape))

