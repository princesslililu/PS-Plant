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


#cd Z:\11. Liat Adler\PS-Plant\Code for detecting circumnutation

#Import Libraries 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.fftpack import fft, rfft, irfft
import glob


#define arrays(?)
xs=[]
ys=[]
zs=[]



#Preparing Control Plots
A = .8
f = 5
t = np.arange(0,1.44,.01)
tophat = np.zeros([144])

for i in range(10):
    tophat[i] = 1
    
plt.plot(t,tophat) 
plt.show()  


x = A*np.cos(2*np.pi*f*t)
#plt.plot(t,x)
plt.axis([0,1,-1,1])
plt.xlabel('time in seconds')
plt.ylabel('amplitude')
#plt.show()

print (x)

#puts together all files that have .txt extension in the working directory
filenames = sorted(glob.glob('*.txt'))
filenames = filenames[0:2]
for f in filenames:
    print(f)


#Open Data

with open('leaf4.txt', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter='\t')   #open the txt file as a csv file, specify that it's tab-delimitiated (\t)
    for row in plots:                            #append the numbers in each column to the x,y, and z arrays. 
        xs.append(float(row[0]))
        ys.append(float(row[1]))
        zs.append(float(row[2]))

#Detrending the Data
x_d=signal.detrend(xs, type == 'constant')  # type can == 'constant' or 'linear' - there doesn't seem to be a difference
y_d=signal.detrend(ys, type == 'constant')
z_d=signal.detrend(zs, type == 'constant')

x_dl=signal.detrend(xs, type == 'linear')
y_dl=signal.detrend(ys, type == 'linear')
z_dl=signal.detrend(zs, type == 'linear')


# Plotting the Detrended Data -  Linear and Constant (to compare)
plt.figure()
plt.plot(x_d, label='x')
plt.plot(y_d, label='y')
plt.plot(10*z_d, label='z') # multiplied by 10 so can visualise better next to x and y
plt.title('Detrended data- constant')
plt.legend(loc='upper right')
plt.show() 
 
plt.figure()
plt.plot(y_dl)
plt.plot(x_dl)
plt.plot(10*z_dl) # multiplied by 10 so can visualise better next to x and y
plt.title('Detrended data - linear')
plt.show()



#setting parameters for FFT, N is equal to the number of points in the time series, not sure about T yet
#y = linspace(a,b,n) generates a row vector y of n points linearly spaced between and including a and b.
N = 144  #needs to be equal to the number of data points in the co-oridnate series data
T = 0.1 / N
d= 1
#x = np.linespace(0.0,  72, 0.5)
#x = np.linespace(0.0, N*T, N) #commented out as not sure this is actually being used


#performing FFT on the detrended data
yf1 = fft(x_d)
yf2 = fft(y_d)
yf3 = fft(z_d)

#performing FFt on the control plots
xff = fft(x)
topff = fft(tophat)


#Return the Discrete Fourier Transform sample frequencies.
# returned float array f contains the frequency bin centers in cycles per unit of the sample spacing 
#(with zero at the start)
# window length n and a sample spacing d
xf = fftfreq(N, T)


#Shift the zero-frequency component to the center of the spectrum.
xf = fftshift(xf)

#Centring FFT about 0 
yplot1 = fftshift(yf1)
yplot2 = fftshift(yf2)
yplot3 = fftshift(yf3)
xplot = fftshift(xff)
topplot = fftshift(topff)

#threshold = 10**2
#idx = np.where(abs(W)>threshold)[0][-1]
#max_f = abs(freq[idx])
#print "Period estimate: ", 1/max_f

#Plot FFT
plt.figure()

plt.plot(xf, np.abs(yplot1), label='x') #include np.abs() around the FFT if want to mod it
plt.plot(xf, np.abs(yplot2), label='y')
plt.plot(xf, np.abs(yplot3), label='z')

#Control plots:
plt.plot(xf, xplot, label='control') # should get 2 peaks 
plt.plot(xf, topplot, label='control top hat') # should get wiggly humps getting smaller

plt.title(f + 'FFT')
plt.legend(loc='upper right')
plt.show()



mY = np.abs(yplot1)
peakY = np.max(mY)
locY = np.argmax(mY)
frqY = xf[locY]

print ('frqY', frqY)

mX = np.abs(yplot2)
peakX = np.max(mX)
locX = np.argmax(mX)
frqX = xf[locX]

print ('frqX', frqX)

mZ = np.abs(yplot3)
peakZ = np.max(mZ)
locZ = np.argmax(mZ)
frqZ = xf[locZ]

print ('frqZ', frqZ)



#def dom (x,y):
#x = 1.0/N * np.abs(yplot)

#plt.figure()
#plt.plot(dom(xf,1.0/N * np.abs(yplot1)
#print the max value in y series
#maxy=max(yplot1)
#max_x = xf[yplot1.argmax(maxy)]  # Find the x value corresponding to the maximum y value
#print (max_x)
#print('max y value is', maxy)

#print ('yf1', yf1)

#print('xf', xf)

#print the max value in y series
maxy = np.max(yplot1)
max_x = xf[yplot1.argmax()]  # Find the x value corresponding to the maximum y value
#print ('max T value for y is', max_x)
#print('max y value is', maxy)

#print the max value in x series
maxx=max(yplot2)
#print('max x value is',maxx)

#print the max value in z series  
maxz=max(yplot3)
#print('max z value is',maxz)




'''
Finding Dominant Frequencey
'''
#y
fourier = np.fft.fft(yf1)
frequencies = np.fft.fftfreq(144, 0.5)  #len() returns the length of a string
#positive_frequencies = frequencies[np.where(frequencies > 1)]  
magnitudes = abs(fourier[np.where(frequencies > 0.9)])

#print('magnitudes for y', magnitudes)


#x
fourier = np.fft.fft(yf2)
frequencies = np.fft.fftfreq(144, 0.5)  #len() returns the length of a string
positive_frequencies = frequencies[np.where(frequencies > 0)]  
magnitudes = abs(fourier[np.where(frequencies > 0)])

#print('magnitudes for x', magnitudes)

#z
fourier = np.fft.fft(yf3)
frequencies = np.fft.fftfreq(144, 0.5)  #len() returns the length of a string
positive_frequencies = frequencies[np.where(frequencies > 0)]  
magnitudes1 = abs(fourier[np.where(frequencies > 0.9)])

print ('magnitudes', magnitudes1)
print ('1/magnitudes', 1/magnitudes1)

#Control
fourier = np.fft.fft(xff)
frequencies = np.fft.fftfreq(144, 0.5)  #len() returns the length of a string
positive_frequencies = frequencies[np.where(frequencies > 0)]  
magnitudes_of_control = abs(fourier[np.where(frequencies > 0.9)])

print ('magnitudes_of_control', magnitudes_of_control)
print ('1/magnitudes', 1/magnitudes_of_control)
# Compute the FFT
W    = np.fft.fft(x_d)   #so I think W here is omega which means there is a facto of 2*pi
freq = np.fft.fftfreq(144,1)

# Look for the longest signal that is "loud"
threshold = 10**2
#idx = np.where(abs(W)>threshold)[0][0]#.nonzero()

#max_f = abs(freq[idx])
#print ("Period estimate: ", 1/max_f)

'''
plt.subplot(211)
plt.scatter([magnitudes1,], [np.abs(yf1[np.where(frequencies > 0.9)]),], s=100,color='r')
plt.plot(freq[:72], abs(W[:72]))
plt.xlabel(r"$f$")

plt.subplot(212)
plt.plot(1.0/freq[:72], abs(W[:72]))
plt.scatter([1/max_f,], [np.abs(W[idx]),], s=100,color='r')
plt.xlabel(r"$1/f$")
plt.xlim(0,20)

plt.show()

'''





#print('magnitudes for z', magnitudes)
#plot
#abs() in Python. The abs() function is used to return the absolute value of a number.
#The argument can be an integer, a floating point number or a complex number.


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



