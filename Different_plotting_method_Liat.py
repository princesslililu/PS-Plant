import matplotlib.pyplot as plt
import numpy as np

x, y, z = np.loadtxt('centroid3.txt', delimiter='\t', unpack=True)
plt.plot(x,y, label='Loaded from file!')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()