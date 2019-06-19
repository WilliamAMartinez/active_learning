#!/usr/bin/env python
# coding: utf-8

# # Implentation of machine learning classification methods in remote sensing
# 
# 
# I want to document the results that the article of Maxwell show. 
# 
# Let's start importing the raster with 220 bands.

# In[124]:


from osgeo import gdal, ogr
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

#Defining work directory
path_wd = os.path.join(os.getcwd(),"dataset/Indian_pines")
raster_file = os.path.join(path_wd,"19920612_AVIRIS_IndianPine_Site3.tif")
source = gdal.Open(raster_file, gdal.GA_ReadOnly)  #Object gdal with the raster info
geo_transform = source.GetGeoTransform()   #matrix with parameters of transformation
proj = source.GetProjectionRef()  #Reference system

#Loop to get all bands from dataset raster and thereafter transfor in array
bands = []
for i in range(1, source.RasterCount + 1):
    band = source.GetRasterBand(i)
    bands.append(band.ReadAsArray())


# In[125]:


#Adding coordinates
x = np.arange(0, 145, 1)
y = np.arange(0, 145, 1)
xx, yy = np.meshgrid(x, y)
bands.append(xx)
bands.append(yy)
#stacking bands
bands_array = np.dstack(bands)
bands_array.shape


# Secondly, let's import the labels for each pixel so we can construc an array of labels with their features.

# In[126]:


raster_label_file = os.path.join(path_wd,"19920612_AVIRIS_IndianPine_Site3_gr.tif")

source_labels = gdal.Open(raster_label_file)
band_labels =  source_labels.GetRasterBand(1)
array_labels = band_labels.ReadAsArray()
array_labels.shape


# Plotting false colour composite blue= band 28, green=  band 65, red = band 128. Stretching information.

# In[127]:


def stretch(image):
    minimum = image.min()
    maximum = image.max()
    image = (image - minimum) / (maximum - minimum)
    return image

bands_array_normal = stretch(bands_array)
plt.imshow(bands_array_normal[:,:,[28,65,128]])


# In[128]:


plt.imshow(array_labels)


# In[129]:


file_labels = os.path.join(path_wd,'19920612_AVIRIS_IndianPine_Site3_gr.txt')
f = open(file_labels,'r')
f1 = f.readlines()
f1


# In[130]:


qnzeros = np.nonzero(array_labels)
Y_init = array_labels[qnzeros]
X_init = bands_array[qnzeros]

#I want to merge the features with repeted labels
dict_labels = {'Alfalfa': [1],
        'Corn': [2,3,4],
        'Grass': [5,7],
        'Hay':[8],
        'Oats':[9],
        'Soybeans':[10,11,12],
        'Wheat':[13],
        'trees':[14]
       }

l_x = np.zeros((1,X_init.shape[1]))
l_y = np.array([0])

for i in dict_labels:
    a = dict_labels[i]
    for j in a:
        index = np.where(Y_init == j)[0]
        l_x = np.concatenate((l_x, X_init[index]))
        l_y = np.concatenate((l_y, np.repeat(i,len(index))))

X = l_x[1:,:-2]
x_coord = l_x[1:,-1]
y_coord = l_x[1:,-2]
Y = l_y[1:]


# In[131]:


[i + ': ' + str(len(np.where(Y == i)[0])) for i in dict_labels]


# Let's split the data into vaidation and test so that leter on we can normalice the dataset. According with the paper we must split data considering a random and stratified selection of samples

# In[132]:


def random_stratified_selection(Y,proportion_training = 0.7):
    '''
    input:
    Y: labels
    proportion_training: percentage associted to the training data
    return:
    index_train, index_test : duple withe indeces for training and validation
    '''
    labels = np.unique(Y)
    index_train = []
    index_valid = []
    for i in labels:
        ind_label = np.where(Y==i)[0]
        number_train = int(len(ind_label)*proportion_training)
        random_all = np.random.choice(ind_label, len(ind_label), replace= False).tolist()
        ind_train = random_all[:number_train]
        ind_val = random_all[number_train:]
        index_train = index_train + ind_train        
        index_valid = index_valid + ind_val

    return index_train, index_valid
        
index_train, index_valid = random_stratified_selection(Y, proportion_training = 0.25)
print(f'number of sampels training: {len(index_train)}')
print(f'number of sampels validation: {len(index_valid)}')
X_train = X[index_train,:]
X_valid = X[index_valid,:]
Y_train = Y[index_train]
Y_valid = Y[index_valid]


# # Random Forest
# 
# Let's start with random forest algorithm. I like to use this algorithm since the tunning of parameters is no completely necesary as long as we play with the default parameters

# In[133]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

classifier= RandomForestClassifier(n_estimators = 500,min_samples_split = 5)
classifier.fit(X_train,Y_train)
#classifying validation data
result_valid = classifier.predict(X_valid)
print('done')


# In[134]:


#results validation
print('Overal Accuracy validation : ', accuracy_score(result_valid, Y_valid))
print('Cohen_kappa validation: ', cohen_kappa_score(result_valid, Y_valid))


# ## Tunning random forest parameters

# In[135]:


'''
split = list(range(2,6))
trees = np.arange(10,300,10)
#loop each element of every parameter
list_accuracy = []
for i in split:
    acc_score = []
    for j in trees:
        classifier_rf = RandomForestClassifier(n_estimators = j,min_samples_split = i)
        classifier_rf.fit(X_train,Y_train)
        result = classifier_rf.predict(X_valid)
        acc_score.append(accuracy_score(result, Y_valid))
    list_accuracy.append(acc_score)
    print(f"done {i}")
'''


# In[136]:


'''
#plotting
#List of colors
list_colors = ['b--','g--','r--','c--','m--','k--','y--']
for k in range(0, len(split)):
    plt.plot(trees,list_accuracy[k],list_colors[k], label = split[k])
plt.ylabel('Accuracy',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Number of trees in the forest',fontsize=15)
plt.legend(title="mtry",fontsize=15)
plt.savefig('sample1.png', bbox_inches="tight")
plt.show()
'''


# # Active learning 

# In[137]:


def entropyfunc(x):
    '''
    input:
    x = vector with the probabities of the classication
    output:
    e = a number expressing the entropy of the vector input
    '''
    x1 = x[x != 0]
    e = -(x1 * np.log2(x1)).sum()
    return e

def softmax(y):
    '''
    input:
    y = vector
    output:
    delta = vector tranformation with an output of porbabilities that sum 1
    '''
    delta = np.exp(y)/np.sum(np.exp(y))
    return delta

def EntropySelection(probabilities):
    #implementing function
    entropy = np.apply_along_axis(entropyfunc, 1, probabilities)
    #fransforming entropie values in probabilities using sof-max function
    entropyvector = softmax(entropy)
    return entropyvector

a = 100
q = 20
n,_= X_train.shape
prob = np.repeat(1/n,n).tolist()
dict_oa = {}
OA_train_list = []
OA_valid_list = []
dict_probabilities = {}

for i in range(a,n,q):
    #weigthed random selection
    l =  np.random.choice(n,i,replace=False,p=prob)
    X_val = X_train[l]
    Y_val = Y_train[l]
    classifier= RandomForestClassifier(n_estimators = 200,min_samples_split = 5)
    classifier.fit(X_val,Y_val)
    #classifying validation and training data
    result_train = classifier.predict(X_train)
    result_valid = classifier.predict(X_valid)
    #probabilities
    train_y_probability = classifier.predict_proba(X_train)
    #scores
    OA_train = accuracy_score(result_train, Y_train)
    OA_valid = accuracy_score(result_valid, Y_valid)    
    #saving data in a dictionary
    OA_train_list.append(OA_train)
    OA_valid_list.append(OA_valid)
    #updating again porbabilities
    prob = EntropySelection(train_y_probability).tolist()
    dict_probabilities[i] = prob
    if i%100 == 0:
        print(f'iteration {i}')

dict_oa["iter"] =  list(range(a,n,q))
dict_oa["train"] = OA_train_list
dict_oa["valid"] =  OA_valid_list

#plot
df = pd.DataFrame.from_dict(dict_oa)


# In[138]:


#visualiztion
import matplotlib.pyplot as plt
ax = plt.gca()

df.plot(kind='line', x ='iter', y = 'train', ax=ax)
df.plot(kind='line', x ='iter', y = 'valid',color='red',  ax=ax)
plt.show()


# # Plotting samples over space

# In[141]:


from geopandas import GeoDataFrame
from shapely.geometry import Point
import rasterio
import rasterio.plot

geometry =  [Point(xy) for xy in zip(y_coord[index_train],x_coord[index_train])]
prob_train_df = pd.DataFrame(dict_probabilities)
prob_train_gdf = GeoDataFrame(prob_train_df,crs =None, geometry = geometry)

raster = rasterio.open(raster_label_file)

fig, ax = plt.subplots(figsize=(8, 8))
rasterio.plot.show(raster, ax=ax)
prob_train_gdf.plot(ax=ax, facecolor='none', edgecolor='red')


# In[253]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def quantile_values(x,q):
    x_percentiles = np.percentile(x,q=q)
    intervals = np.array(range(len(x_percentiles)-1))
    # empty array
    markersize_points = np.zeros(len(x))
    #loop every element of the array
    for i in intervals:
        index = np.where((x >= x_percentiles[i]) & (x < x_percentiles[i + 1]))[0]
        markersize_points[index] = np.exp(i)
    return markersize_points

#Loop for the markers
markersize = {}
for k in range(0,prob_train_gdf.shape[1]-1):
    x = prob_train_gdf.iloc[:,k].values
    q = np.array([0,20,40,60,80,100])
    markersize[k] = quantile_values(x,q)

#let's use matplotlib to map this time the points

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    ax1.clear()
    ax1.scatter(y_coord[index_train], x_coord[index_train], s = markersize[i])
    print(i)
    return ax1

ani = FuncAnimation(fig, animate, frames = list(range(0,100)), interval=500)
plt.show()
print('done')


# In[254]:


list(range(0,4))


# In[255]:


fig = plt.figure()
plt.scatter(x_coord[index_train], y_coord[index_train], s = markersize[0])
plt.show()


# In[198]:


prob_train_gdf.shape[1]


# In[ ]:




#fig, ax = plt.subplots(figsize = (8,8))
#plt.scatter(y_coord[index_train],x_coord[index_train],c='red', s = markersize_points)
#plt.show()

#fig, ax = plt.subplots(figsize = (10,8))
#plt.hist2d(y_coord[index_train],x_coord[index_train], (40, 40), cmap=plt.cm.jet,weights = markersize_points)
#plt.colorbar()

#fig, ax = plt.subplots(figsize = (8,8))
#prob_train_gdf.plot(marker = 'o',color = 'red',markersize = markersize_points,ax=ax)


# In[ ]:




