import matplotlib.pyplot as plt
import numpy as np
import scipy

# Create data
mean = 0
std = 1

funcX = np.arange(-5, 5, 0.1)
funcY = 1 / ((2*np.pi)**0.5 * std) * np.exp(-(funcX-mean)**2 / (2*std**2))

data = np.random.normal(mean, std, 1000)
# count, bins, ignored = plt.hist(data, 30, density=True)


# Create Figure
fig = plt.figure(num=100, figsize=(4,3), tight_layout=True)
ax = plt.axes()

ax.plot(funcX, funcY, color='coral', linewidth=3.0, linestyle='-', label='L1')
ax.hist(data, 30, density=True, color='teal')
ax.set_xlim((-5, 5))
ax.set_ylim((0, 0.5))

ax.set_xticks(np.linspace(-5, 5, 11))
ax.set_yticks(np.linspace(0, 0.5, 5))

ax.set_xlabel('Values of Random Variable $x$', family='Arial', fontstyle='italic', fontsize=12)
ax.set_ylabel('Probability', family='Arial', fontstyle='italic', fontsize=12)

ax.set_title('Normal Distribution, $\mu = 0$, $\sigma = 1$', family='Arial', fontsize=14)
ax.grid(color='0.2', linestyle='-', linewidth=0.1)

plt.show()