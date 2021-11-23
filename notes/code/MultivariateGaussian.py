import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np


def ComputePdfForMultiGauss(x1, x2, mu, sigma):
    pdf = np.zeros((len(x1), len(x2)))
    for itr1 in range(len(x1)):
        for itr2 in range(len(x2)):
            coor = np.array((x1[itr1], x2[itr2]))
            coor = coor.reshape((2,1))

            delta = (coor-mu)
            sigmaInv = np.linalg.inv(sigma)
            sigmaDet = np.linalg.det(sigma)

            pdf[itr1, itr2] = 1/(2*np.pi*sigmaDet**0.5) * np.exp(-1/2 * (delta.T).dot(sigmaInv).dot(delta))

    xx, yy = np.meshgrid(x1, x2)
    return xx, yy, pdf


def AddAxe(fig, axIdx, xx, yy, pdf):
    ax = fig.add_subplot(1, 3, axIdx, projection='3d', proj_type='ortho')
    ax.plot_surface(xx, yy, pdf, rstride=1, cstride=1, cmap=cm.plasma, linewidth=0, antialiased=False)
    ax.set_xlabel('$x_1$', family='Arial', fontstyle='italic', fontsize=12)
    ax.set_ylabel('$x_2$', family='Arial', fontstyle='italic', fontsize=12)
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    ax.set_title('$\mu_{:1d}, \Sigma_{:1d}$'.format(axIdx, axIdx))
    ax.set_zticklabels('')
    ax.view_init(90, 90)


# Range of x1 and x2
x1 = np.linspace(-4, 4, 120)
x2 = np.linspace(-4, 4, 120)

# Case 1
mu1 = np.zeros([2, 1])
sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
xx1, yy1, pdf1 = ComputePdfForMultiGauss(x1, x2, mu1, sigma1)

# Case 2
mu2 = np.zeros([2, 1])
sigma2 = np.array([[1.0, 0.5], [0.5, 1.0]])
xx2, yy2, pdf2 = ComputePdfForMultiGauss(x1, x2, mu2, sigma2)

# Case 1
mu3 = np.zeros([2, 1])
sigma3 = np.array([[1.0, -0.5], [-0.5, 1.0]])
xx3, yy3, pdf3 = ComputePdfForMultiGauss(x1, x2, mu3, sigma3)


# Create Figure
fig = plt.figure(figsize=(6.6, 2.8), tight_layout=True)
AddAxe(fig, 1, xx1, yy1, pdf1)
AddAxe(fig, 2, xx2, yy2, pdf2)
AddAxe(fig, 3, xx3, yy3, pdf3)
plt.show()
