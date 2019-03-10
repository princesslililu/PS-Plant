import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
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
print (xs)
print (ys)
print (zs)
plt.plot(xs,ys,zs)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(xs, ys, zs, label='parametric curve')

plt.figure()

