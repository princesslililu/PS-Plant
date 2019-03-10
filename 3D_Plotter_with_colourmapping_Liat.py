import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
from scipy import signal
xs=[]
ys=[]
zs=[]


with open('centroid3.txt', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter='\t')
    for row in plots:
        xs.append(float(row[0]))
        ys.append(float(row[1]))
        #x1.append(signal.detrend(x))
        zs.append(float(row[2]))
#Axes3D.plot(xs, ys, zs)

x1=signal.detrend(xs)
print (x1)
print (xs)

N_points = 144
x1 = np.arange(N_points, dtype=float)
t = x1

fig = plt.figure()
ax = fig.gca(projection='3d')
t /= max(t)
for i in range(1, N_points):
    ax.plot(x1[i-1:i+1], ys[i-1:i+1], zs[i-1:i+1],c=(t[i-1], 0, 0), label='parametric curve')
print (xs)
print (ys)
print (zs)
#plt.plot(xs[i-1:i+1], ys[i-1:i+1], zs[i-1:i+1], c=(xs[i-1], 0, 0))
plt.show()



#N_points = 10
#x = np.arange(N_points, dtype=float)
#t = x

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#t /= max(t)
#for i in range(1, N_points):
    #ax.plot(xs[i-1:i+1], ys[i-1:i+1], zs[i-1:i+1],c=(t[i-1], 0, 0), label='parametric curve')

#plt.figure()
#fig.show()


#Axes3D.show()