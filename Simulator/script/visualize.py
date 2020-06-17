import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils  
#import lineGenerator

# set the boundary of the 3D space
x1 = -10.
x2 = 10.
y1 = 0.
y2 = 15.
z1 = 0.
z2 = 3.5

nptype = np.float64
fig = plt.figure()
ax = fig.gca(projection='3d')

utils.visualizeLines3D(ax)
utils.visualizeStars3D(ax)
utils.visualizeTriangles3D(ax)
utils.visualizeCubes3D(ax)
utils.visualizeTrajactory3D(ax)

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#ax.set_xlim(x1, x2)
#ax.set_ylim(y1, y2)
#ax.set_zlim(z1, z2)
plt.show() 
