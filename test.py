import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d

ax = plt.figure().add_subplot(projection='3d')


def Sigmoid(x):
       return 1/(1+np.exp(-x))

def Relu(x):
       return np.maximum(0,x)
def MSE(y,t):
       return (y - t)**2/np.size(y)#t是正确的标签
def CEE(y,t):
       return -np.sum(t*np.log(y)+(1-t)*np.log(1-y))/len(y)##y的值在0-1之间


def Get_Test_Data(delta):
       x = y = np.arange(-3.0, 3.0, delta)
       X, Y = np.meshgrid(x, y)

       W1 = np.array([1,-3,5]).reshape(3, 1, 1)
       W11= np.array([0.2,4,6]).reshape(3, 1, 1)
       b1 = np.array([0.1,0.2,0.3]).reshape(3, 1, 1)
       W2 = np.array([[1,-4],[2,5],[3,6]])
       b2 = np.array([1,0.2])
       W3 = np.array([2.0,-1])
       b3 = np.array([0.2])


       a1 = W1*X+W11*Y+b1
       Z1 = Relu(a1)
       a2 = np.tensordot(W2, Z1, axes=(0, 0)) + b2.reshape(-1, 1, 1)
       Z2 =  Relu(a2)

       a3 = np.tensordot(W3, Z2, axes=(0, 0)) + b3
       Z =  Relu(a3)
       return X, Y, Z




X, Y, Z = Get_Test_Data(0.5)

# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                alpha=0.3)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
ax.contourf(X, Y, Z, zdir='z', offset=-10, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='x', offset=-4, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='y', offset=4, cmap='coolwarm')

ax.set(xlim=(-4, 4), ylim=(-4, 4), zlim=(-10,10),
       xlabel='X', ylabel='Y', zlabel='Z')

plt.show()



