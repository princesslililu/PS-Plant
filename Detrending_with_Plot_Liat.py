#import libs and wheels needed

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
from scipy import signal

#define arrays(?)
xs=[]
ys=[]
zs=[]

#open the txt file as a csv file, specify that it's tab-delimitered
#append the numbers in each column to the x,y, and z arrays. 
with open('centroid3.txt', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter='\t')
    for row in plots:
        xs.append(float(row[0]))
        ys.append(float(row[1]))
        zs.append(float(row[2]))


#detrend  data series 
x1=signal.detrend(xs, type =='constant')
print (x1)
#print (xs)

plt.plot(x1,label='raw')
plt.plot(xs,label='detrended')
plt.title('Comparison before and after Detrending')
plt.legend(loc='middle left')
plt.xlabel('t')
plt.ylabel('x-coordinate value')
plt.show()