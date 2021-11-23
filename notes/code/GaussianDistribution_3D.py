import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import numpy as np

# Create data-1
mu1 = 7
sigma1 = 1.5
data1 = np.random.normal(mu1, sigma1, 40)

# Create data-2
mu2 = 5
sigma2 = 1
data2 = np.random.normal(mu2, sigma2, 40)

# Create function-1
x1 = np.arange(0, 10, 0.2)
z1 = 1 / ((2*np.pi)**0.5 * sigma1) * np.exp(-(x1-mu1)**2 / (2*sigma1**2))

x2 = np.arange(0, 10, 0.2)
z2 = 1 / ((2*np.pi)**0.5 * sigma2) * np.exp(-(x2-mu2)**2 / (2*sigma2**2))

xx, yy = np.meshgrid(x1, x2)
zz = 1 / ((2*np.pi)**0.5 * sigma1) * np.exp(-(xx-mu1)**2 / (2*sigma1**2)) * 1 / ((2*np.pi)**0.5 * sigma2) * np.exp(-(yy-mu2)**2 / (2*sigma2**2))

# Create Figure
fig = plt.figure(figsize=(6,6), tight_layout=True)

# ax1
ax = fig.add_subplot(2, 2, 1)
ax.axhline(y=mu2, xmin=0, xmax=1, color='silver', linewidth=1.0, linestyle='--')
ax.axvline(x=mu1, ymin=0, ymax=1, color='silver', linewidth=1.0, linestyle='--')
ax.plot(data1, data2, color='k', linewidth=3.0, linestyle='', marker='o', markerfacecolor='w', label='L1')
ax.set_xlim((0, 10))
ax.set_ylim((0, 10))
ax.set_xlabel('$x_1$', family='Arial', fontstyle='italic', fontsize=12)
ax.set_ylabel('$x_2$', family='Arial', fontstyle='italic', fontsize=12)

# ax2
ax = fig.add_subplot(2, 2, 2)
ax.plot(z2, x2, color='k', linewidth=1.0, linestyle='-', label='L1')
ax.axhline(y=mu2, xmin=0, xmax=1, color='silver', linewidth=1.0, linestyle='--')
ax.set_ylim((0, 10))
ax.set_xlim((0, 0.5))
ax.set_ylabel('$x_2$', family='Arial', fontstyle='italic', fontsize=12)
ax.set_xlabel('$p(x_2)$', family='Arial', fontstyle='italic', fontsize=12)

# ax3
ax = fig.add_subplot(2, 2, 3)
ax.plot(x1, z1, color='k', linewidth=1.0, linestyle='-', label='L1')
ax.axvline(x=mu1, ymin=0, ymax=1, color='silver', linewidth=1.0, linestyle='--')
ax.set_xlim((0, 10))
ax.set_ylim((0, 0.5))
ax.set_xlabel('$x_1$', family='Arial', fontstyle='italic', fontsize=12)
ax.set_ylabel('$p(x_1)$', family='Arial', fontstyle='italic', fontsize=12)

# ax4
ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('$x_1$', family='Arial', fontstyle='italic', fontsize=12)
ax.set_ylabel('$x_2$', family='Arial', fontstyle='italic', fontsize=12)
ax.set_zticklabels('')
ax.view_init(20, -60)
plt.show()