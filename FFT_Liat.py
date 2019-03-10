#from scipy.fftpack import fft
# Number of sample points
#N = 600
# sample spacing
#T = 1.0 / 800.0
##x = np.linspace(0.0, N*T, N)
#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
#yf = fft(y)
#xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
#import matplotlib.pyplot as plt
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
#plt.grid()
#plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.fftpack import fft, rfft, irfft

#define arrays(?)
xs=[]
ys=[]
zs=[]

#open the txt file as a csv file, specify that it's tab-delimitiated
#append the numbers in each column to the x,y, and z arrays. 
with open('centroid3.txt', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter='\t')
    for row in plots:
        xs.append(float(row[0]))
        ys.append(float(row[1]))
        zs.append(float(row[2]))

#detrending the data
x_d=signal.detrend(xs, type == 'constant')
y_d=signal.detrend(ys, type == 'constant')
z_d=signal.detrend(zs, type == 'constant')

x_dl=signal.detrend(xs, type == 'linear')
y_dl=signal.detrend(ys, type == 'linear')
z_dl=signal.detrend(zs, type == 'linear')

#plotting the detrended data
plt.figure()
plt.plot(y_d)
plt.plot(x_d)
plt.plot(z_d)
plt.title('Detrended data- constant')
#plt.show()

plt.figure()
plt.plot(y_dl)
plt.plot(x_dl)
plt.plot(z_dl)
plt.title('Detrended data - linear')


N = 144
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)

#setting the input data
yx = x_d
yy = y_d
yz = z_d

#performing FFT 
yf1 = fft(yx)
yf2 = fft(yy)
yf3 = fft(yz)
xf = fftfreq(N, T)
xf = fftshift(xf)

#preparing plot?
yplot1 = fftshift(yf1)
yplot2 = fftshift(yf2)
yplot3 = fftshift(yf3)



#plot
plt.figure()
plt.plot(xf, 1.0/N * np.abs(yplot1))
plt.plot(xf, 1.0/N * np.abs(yplot2))
plt.plot(xf, 1.0/N * np.abs(yplot3))
plt.title('FFT of  coordinate series')

# Tweaking display region and labels
#ax.set_xlim(0, 10)
#ax.set_ylim(0, 10)
#ax.set_zlim(0, 10)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.grid()
plt.show()



from scipy.optimize import least_squares

y_d_ls = least_squares(yplot, (0.1, 0.1), bounds=([0, 0], [1, 1]))
plt.figure()
plt.plot(y_d_ls)



