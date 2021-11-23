import matplotlib.pyplot as plt
import numpy as np

##################################################################
# Functions
##################################################################
def ExpandPolynomial(m, n, x):
    x1 = np.ones((m,n+1))
    for j in np.linspace(1,n,n):
        power = int(j)
        x1[:,power] = np.power(x, power)

    return x1


def ComputeCost(x, y, theta, m, lmbda):
    hypth = x.dot(theta)
    delta = hypth - y.reshape((m,1))

    theta2 = np.power(theta, 2)

    cost =  1 / (2*m) * ( delta.T.dot(delta)[0,0] + lmbda * theta2[1:].sum() )
    return cost


def ComputeGradient(x, y, theta, lmbda, m, n):
    xT = x.T
    h = x.dot(theta)
    delta = h - y.reshape((m,1))
    filter = np.eye(n+1)
    filter[0,0] = 0

    grad = 1 / m * (xT.dot(delta) + lmbda * filter.dot(theta))
    return grad


def IterateGradientDescent(x, y, theta, lmbda, alpha, m, n, epsilon=1e-20, maxIterate=2000000):
    itr = 0
    costHistory = []
    converge = False
    
    while not converge and itr < maxIterate:
        cost = ComputeCost(x, y, theta, m, lmbda)
        costHistory.append(cost)

        grad = ComputeGradient(x, y, theta, lmbda, m, n)
        theta -= alpha * grad
        
        if itr > 0:
            costPre = costHistory[itr-1]
            costDelta = abs(cost - costPre)

            if costDelta < epsilon:
                converge = True

        itr += 1

    itrHistory = np.linspace(0, itr-1, itr)

    return theta, itrHistory, costHistory


def AddPlot(ax, dataX, dataY, lmbda, lineOrMarker, axisScale, lineColor):
    if lineOrMarker == 'marker':
        ax.plot(dataX, dataY, color='k', linewidth=0.0, linestyle='', marker='o', markerfacecolor='w', label='training')
    else:
        if axisScale == 'linear':
            ax.plot(dataX, dataY, color=lineColor, linewidth=1.0, linestyle='--', label='$\lambda$ = {}'.format(lmbda))
        else:
            ax.loglog(dataX, dataY, color=lineColor, linewidth=1.0, linestyle='--', label='$\lambda$ = {}'.format(lmbda))


def SetPlot(ax, xlabel, ylabel, title, xlim=None, ylim=None):
    ax.set_xlabel(xlabel, family='Arial', fontstyle='italic', fontsize=12)
    ax.set_ylabel(ylabel, family='Arial', fontstyle='italic', fontsize=12)
    ax.set_title(title, family='Arial', fontstyle='italic', fontsize=12)
    
    if xlim != None:
        ax.set_xlim(xlim)
    
    if ylim != None:
        ax.set_ylim(ylim)
    
    ax.legend(loc='upper left')


##################################################################
# main
##################################################################
# Data
trainingSet = np.array([[0.1, 0.1], 
                        [0.3, 0.7],
                        [0.5, 0.8],
                        [0.6, 1.2],
                        [0.8, 1.0],
                        [0.9, 1.1]])


xOri = trainingSet[:,0]
y = trainingSet[:,1]


# Model settings
m = len(xOri)
n = 5
lmbdaList = [0, 5e-6]
alpha = 1.0
# theta = np.ones((n+1,1)) * 10
theta = np.array([[   -8.03],
                  [   80.15],
                  [ -501.56],
                  [ 1000.48],
                  [-1000.07],
                  [  200.61]])


# Iteration terminated condition
epsilon = 1e-20
maxIterate = 2000000


# Figure settings
lineColorList = ['orange', 'deepskyblue']
figH = plt.figure(figsize=(4, 3), tight_layout=True)
axH = figH.add_subplot(1, 1, 1)

figJ = plt.figure(figsize=(4, 3), tight_layout=True)
axJ = figJ.add_subplot(1, 1, 1)


# Main procedures
for idx, lmbda in enumerate(lmbdaList):
    # Gradient descent
    xPoly = ExpandPolynomial(m, n, xOri)
    thetaImp, itrHistory, costHistory = IterateGradientDescent(xPoly, y, theta, lmbda, alpha, m, n, epsilon=epsilon, maxIterate=maxIterate)


    # Normal equation method
    xTx = xPoly.T.dot(xPoly)
    xTxInv = np.linalg.pinv(xTx)
    thetaExp = xTxInv.dot(xPoly.T).dot(y)


    # Hyperthesis
    xTest = np.linspace(0.0, 1.0, 100)
    xTestPoly = ExpandPolynomial(100, n, xTest)
    hImp = xTestPoly.dot(thetaImp)
    hExp = xTestPoly.dot(thetaExp)


    # Plot
    if idx == 0:
        AddPlot(axH, xOri, y, lmbda=None, lineOrMarker='marker', axisScale='linear', lineColor=None)

    color = lineColorList[idx]
    AddPlot(axH, xTest, hImp, lmbda, lineOrMarker='line', axisScale='linear', lineColor=color)
    AddPlot(axJ, itrHistory, costHistory, lmbda, lineOrMarker='line', axisScale='loglog', lineColor=color)


SetPlot(axH, 'x', 'h(x)', 'n = {}'.format(n), [0.0, 1.0], [0.0, 1.5])
SetPlot(axJ, 'iter', 'cost', 'n = {}'.format(n))
plt.show()