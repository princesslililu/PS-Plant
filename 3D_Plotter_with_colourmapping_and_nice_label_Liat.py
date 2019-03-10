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
#Axes3D.plot(xs, ys, zs)

#detrend  data series 
#detrended_data_set=signal.detrend(xs, ys, zs)
print (x1)
print (xs)

#Setting up the colour mapping: set N_points to the numer of data points in the time series
N_points = 144
x1 = np.arange(N_points, dtype=float)
t = x1

#Plotting the figure with colour mapping
fig = plt.figure()
ax = fig.gca(projection='3d')
t /= max(t)
for i in range(1, N_points):
    ax.plot(x1[i-1:i+1], ys[i-1:i+1], zs[i-1:i+1],c=(t[i-1], 0, 0))
    
plt.title('leaf motion test with colour mapping')

# Tweaking display region and labels
#ax.set_xlim(0, 10)
#ax.set_ylim(0, 10)
#ax.set_zlim(0, 10)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')



#output the values used to make graph just to check
print (xs)
print (ys)
print (zs)

plt.show()



