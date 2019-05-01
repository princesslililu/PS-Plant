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


#import libraries 
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

#plotting the detrended data - both the constant and linear ones to compare
plt.figure()
plt.plot(x_d, label='x')
plt.plot(y_d, label='y')
plt.plot(10*z_d, label='z') # multiplied by 10 so can visualise better next to x and y
plt.title('Detrended data- constant')
plt.legend(loc='upper right')
#plt.show()
 
plt.figure()
plt.plot(y_dl)
plt.plot(x_dl)
plt.plot(10*z_dl) # multiplied by 10 so can visualise better next to x and y
plt.title('Detrended data - linear')

plt.show()

#setting parameters for FFT, N is equal to the number of points in the time series, not sure about T yet
#y = linspace(a,b,n) generates a row vector y of n points linearly spaced between and including a and b.
N = 144
T = 1.0 / 144.0
d= 1
#x = np.linespace(0.0,  72, 0.5)
#x = np.linespace(0.0, N*T, N) #commented out as not sure this is actually being used


#performing FFT on the detrended data
yf1 = fft(x_d)
yf2 = fft(y_d)
yf3 = fft(z_d)

#Return the Discrete Fourier Transform sample frequencies.
# returned float array f contains the frequency bin centers in cycles per unit of the sample spacing 
#(with zero at the start)
# window length n and a sample spacing d
xf = fftfreq(N, T)

#Shift the zero-frequency component to the center of the spectrum.
xf = fftshift(xf)

#preparing plot?
yplot1 = fftshift(yf1)
yplot2 = fftshift(yf2)
yplot3 = fftshift(yf3)
plt.show()
#print the max value in y series
#maxy=max(yplot1)
#max_x = xf[yplot1.argmax(maxy)]  # Find the x value corresponding to the maximum y value
#print (max_x)
#print('max y value is', maxy)


#print the max value in y series
maxy=max(yplot1)
max_x = xf[yplot1.argmax()]  # Find the x value corresponding to the maximum y value
print ('max T value for y is', max_x)
print('max y value is', maxy)

#print the max value in x series
maxx=max(yplot2)
print('max x value is',maxx)

#print the max value in z series  
maxz=max(yplot3)
print('max z value is',maxz)





#finding dominant frequencey
fourier = np.fft.fft(yf1)
frequencies = np.fft.fftfreq(144, 0.5)  #len() returns the length of a string
#positive_frequencies = frequencies[np.where(frequencies > 1)]  
magnitudes = abs(fourier[np.where(frequencies > 0.9)])

print('magnitudes for y', magnitudes)


#x
fourier = np.fft.fft(yf2)
frequencies = np.fft.fftfreq(144, 0.5)  #len() returns the length of a string
positive_frequencies = frequencies[np.where(frequencies > 0)]  
magnitudes = abs(fourier[np.where(frequencies > 0)])

print('magnitudes for x', magnitudes)

#z
fourier = np.fft.fft(yf3)
frequencies = np.fft.fftfreq(144, 0.5)  #len() returns the length of a string
positive_frequencies = frequencies[np.where(frequencies > 0)]  
magnitudes = abs(fourier[np.where(frequencies > 0)])

print('magnitudes for z', magnitudes)
#plot
#abs() in Python. The abs() function is used to return the absolute value of a number.
#The argument can be an integer, a floating point number or a complex number.

plt.figure()
plt.plot(xf, 1.0/N * np.abs(yplot1), label='x')       
plt.plot(xf, 1.0/N * np.abs(yplot2), label='y')
plt.plot(xf, 1.0/N * np.abs(yplot3), label='z')
plt.title('FFT of  coordinate series')
plt.legend(loc='upper right')
plt.show()
#def dom (x,y):
#x = 1.0/N * np.abs(yplot)

plt.figure()
#plt.plot(dom(xf,1.0/N * np.abs(yplot1)

# Tweaking display region and labels
#ax.set_xlim(0, 10)
#ax.set_ylim(0, 10)
#ax.set_zlim(0, 10)
#ax.set_xlabel('X axis')
#ax.set_ylabel('Y axis')
#ax.set_zlabel('Z axis')

#plt.grid()


#extracting the dominant frequencey:

#import numpy
#from numpy import sin
#from math import pi

#Fs=200.
#F=50.
#t = [i*1./Fs for i in range(200)]
#y = sin(2*pi*numpy.array(t)*F)

#fourier = numpy.fft.fft(y)
#frequencies = numpy.fft.fftfreq(len(t), 0.005)  # where 0.005 is the inter-sample time difference
#positive_frequencies = frequencies[numpy.where(frequencies > 0)]  
#magnitudes = abs(fourier[numpy.where(frequencies > 0)])  # magnitude spectrum

#peak_frequency = numpy.argmax(magnitudes)

#from scipy.optimize import least_squares

#y_d_ls = least_squares(yplot, (0.1, 0.1), bounds=([0, 0], [1, 1]))
#plt.figure()
#plt.plot(y_d_ls)



