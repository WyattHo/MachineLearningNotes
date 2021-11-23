import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits import mplot3d
import numpy as np


fig = plt.figure(figsize=plt.figaspect(0.5))

# Make data.
X = np.arange(0, 6, 0.1)
Y = np.arange(2, 8, 0.1)
X, Y = np.meshgrid(X, Y)

dist = (X-3)**2 + (Y-5)**2
sigmaList = [0.5, 1.0, 1.5]

for idx, sigma in enumerate(sigmaList):
    Z = np.exp(-dist/(2*sigma**2))
    
    # Plot the surface.
    ax = fig.add_subplot(2, 3, idx+1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    
    ax.set_title('$\sigma = {}$'.format(sigma))
    
    # Customize the z axis.
    ax.set_xlim(0., 6.)
    ax.set_ylim(2., 8.)
    ax.set_zlim(0., 1.)

    # Plot the contour
    ax = fig.add_subplot(2, 3, idx+4, projection='3d')
    surf = ax.contour(X, Y, Z, cmap=cm.coolwarm)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.set_zticklabels('')
    ax.view_init(90, 90)
    

plt.show()
