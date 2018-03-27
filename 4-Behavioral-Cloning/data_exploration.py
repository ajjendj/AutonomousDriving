# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 21:51:31 2017

@author: ajjenjoshi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import misc

df = pd.read_csv('data/driving_log.csv')
#s = df.steering.values
#plt.hist(s, normed=False, bins=100, color='blue')
#plt.xlabel('Steering Angles')
#plt.ylabel('Counts')
#plt.title('Histogram of steering values of data')
#
#s_balanced = []
#for i in range(s.shape[0]):
#    if s[i] ==  0:
#        if np.random.random() < 0.1:
#            s_balanced.append(s[i])
#    else:
#        s_balanced.append(s[i])
#s_balanced = np.asarray(s_balanced)
#            
#plt.hist(s_balanced, normed=False, bins=100, color='blue')
#plt.xlabel('Steering Angles')
#plt.ylabel('Counts')
#plt.title('Histogram of data after balancing samples with 0 steering angle')       

image = misc.imread('data/'+'IMG/center_2016_12_01_13_35_30_724.jpg')
image_left = misc.imread('data/'+'IMG/left_2016_12_01_13_35_30_724.jpg')
image_right = misc.imread('data/'+'IMG/center_2016_12_01_13_35_30_724.jpg')
plt.imshow(image_right)
plt.title('Image from right camera, Steering = -.20')