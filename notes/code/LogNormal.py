import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

muList    = [  0,   0,   0,   0,   0,   0]
sigmaList = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

colorOrigin = (0.8, 0.8, 0.8)

# Create Figure
fig, axs = plt.subplots(2, 2, figsize=(6,5), tight_layout=True)


for itr in range(len(muList)):
    mu = muList[itr]
    sigma = sigmaList[itr]

    color = tuple([comp-itr/len(muList)*comp for comp in colorOrigin])

    x = np.arange(0.01, 10, 0.01)
    pdf = 1 / (x * sigma * (2*np.pi)**0.5 ) * np.exp(-(np.log(x)-mu)**2 / (2*sigma**2))
    cdf = 1/2 * (1 + erf( (np.log(x) - mu) / ( sigma * 2**0.5) ) )

    axs[0, 0].plot(x, pdf, color=color, linewidth=1.0, linestyle='-')
    axs[0, 1].semilogx(x, pdf, color=color, linewidth=1.0, linestyle='-')
    axs[1, 0].plot(x, cdf, color=color, linewidth=1.0, linestyle='-')
    axs[1, 1].semilogx(x, cdf, color=color, linewidth=1.0, linestyle='-')


axs[0, 0].set_xlim((0, 2.5))
axs[0, 0].set_ylim((0, 1))
axs[0, 0].set_xticks(np.linspace(0, 2.5, 6))
axs[0, 0].set_yticks(np.linspace(0, 1, 5))
axs[0, 0].set_ylabel('Probability density', family='Arial', fontstyle='italic', fontsize=12)
axs[0, 0].grid(color='0.2', linestyle='-', linewidth=0.1)

axs[0, 1].set_xlim((0.01, 10))
axs[0, 1].set_ylim((0, 1))
axs[0, 1].set_yticks(np.linspace(0, 1, 5))
axs[0, 1].grid(color='0.2', linestyle='-', linewidth=0.1)

axs[1, 0].set_xlim((0, 2.5))
axs[1, 0].set_ylim((0, 1))
axs[1, 0].set_xticks(np.linspace(0, 2.5, 6))
axs[1, 0].set_yticks(np.linspace(0, 1, 5))
axs[1, 0].set_ylabel('Cumulative distribution', family='Arial', fontstyle='italic', fontsize=12)
axs[1, 0].grid(color='0.2', linestyle='-', linewidth=0.1)

axs[1, 1].set_xlim((0.01, 10))
axs[1, 1].set_ylim((0, 1))
axs[1, 1].set_yticks(np.linspace(0, 1, 5))
axs[1, 1].grid(color='0.2', linestyle='-', linewidth=0.1)

plt.show()