# This is a 3D landmark generator to build 3D features and
# basically they are lines. Basic shapes should be thoughtfully
# input and it will generate 3D landmarks accordingly using
# arbitrarily set transform matrix to reshape the objects and
# move they to the wanted location.

# INFO: Before running the .py file, please restore the previous generated data or they will be missed and replaced!
#       There is a visualize.py file to see it.

import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils  

nptype = np.float64
fig = plt.figure()
ax = fig.gca(projection='3d')

# set the boundary of the 3D space
x1 = -6.
x2 = 6.
y1 = -6.
y2 = 6.
z1 = -6.
z2 = 6.

# set the basic shape
# 1) lines
l0 = np.array([[0., 1., 0.],[0., 0., 0.]],dtype = nptype)
# 2) stars
s0 = np.array([[0., 0., 0.],[1., 0., 0.],[-1., 0., 0.],[0., 1., 0.],[0., -1., 0.],
              [0., 0., 1.],[0., 0., -1.]],dtype = nptype)
# 3) triangles
t0 = np.array([[0., 1., 0.],[0.5, 0., 0.],[-0.5, 0., 0.]],dtype = nptype)
# 4) cubes
c0 = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],[1., 1., 0.],
               [0., 0., 1.], [1., 0., 1.], [0., 1., 1.],[1., 1., 1.]],dtype = nptype)

# common functions for landmarks building
PI = 3.141692654
lines = np.array([],dtype = nptype) 
stars = np.array([],dtype = nptype) 
triangles = np.array([],dtype = nptype) 
cubes = np.array([],dtype = nptype)

def landmarksBuilding(devide, _x1, _x2, _y1, _y2, _z1, _z2):
  #devide: how many landmarks you want to draw on a certain direction
  #x, y, z: the range of translation that the landmarks are placed
  #this function only draws basic images: lines, stars, triangles, cubes
  global lines, stars, triangles, cubes
  for yi in range(devide):
    #rand_scaler = random.uniform(0.75, (y2-y1)/devide)
    rand_choose = random.randint(1,4)
    rand_angle1 = random.uniform(-PI, PI)
    rand_angle2 = random.uniform(-PI, PI)
    rand_angle3 = random.uniform(-PI, PI)
    rand_trans1 = random.uniform(_x1, _x2)
    rand_trans2 = random.uniform(_y1, _y2)
    rand_trans3 = random.uniform(_z1, _z2)
    if (rand_choose == 1):
      #l_temp = utils.scale(0.75, (y2-y1)/devide-0.2, l0)
      l_temp = utils.affine(0.6, 0.7, l0)
      l_temp = utils.multi(utils.EulerRotate(rand_angle1, rand_angle2, rand_angle3),l_temp)
      #l_temp = utils.translate(rand_trans1, rand_trans2+(yi+0.5)*(y2-y1)/devide, rand_trans3, l_temp)
      l_temp = utils.translate(rand_trans1, rand_trans2, rand_trans3, l_temp)
      lines = np.append(lines, l_temp)
    elif (rand_choose == 2):
      #s_temp = utils.scale(0.75, (y2-y1)/devide-0.2, s0)
      s_temp = utils.affine(0.6, 0.7, s0)
      s_temp = utils.multi(utils.EulerRotate(rand_angle1, rand_angle2, rand_angle3),s_temp)
      #s_temp = utils.translate(rand_trans1, rand_trans2+(yi+0.5)*(y2-y1)/devide, rand_trans3, s_temp)
      s_temp = utils.translate(rand_trans1, rand_trans2, rand_trans3, s_temp)
      stars = np.append(stars, s_temp)
    elif (rand_choose == 3):
      #t_temp = utils.scale(0.75, (y2-y1)/devide-0.2, t0)
      t_temp = utils.affine(0.6, 0.7, t0)
      t_temp = utils.multi(utils.EulerRotate(rand_angle1, rand_angle2, rand_angle3),t_temp)
      #t_temp = utils.translate(rand_trans1, rand_trans2+(yi+0.5)*(y2-y1)/devide, rand_trans3, t_temp)
      t_temp = utils.translate(rand_trans1, rand_trans2, rand_trans3, t_temp)
      triangles = np.append(triangles, t_temp)
    elif (rand_choose == 4):
      #c_temp = utils.scaleCubes(0.75, (y2-y1)/devide-0.2, c0)
      c_temp = utils.affine(0.6, 0.7, c0)
      c_temp = utils.multi(utils.EulerRotate(rand_angle1, rand_angle2, rand_angle3),c_temp)
      #c_temp = utils.translate(rand_trans1, rand_trans2+(yi+0.5)*(y2-y1)/devide, rand_trans3, c_temp)
      c_temp = utils.translate(rand_trans1, rand_trans2, rand_trans3, c_temp)
      cubes = np.append(cubes, c_temp)
    else:
      print('Warning: Error Occurs in right!')

def manMadeLandmarks():
  global lines, stars, triangles, cubes, x1, y1, z1, x2, y2, z2
  #frame
  cubes = np.append(cubes, np.array([[x1, y1, z1], [x2, y1, z1], [x1, y2, z1],[x2, y2, z1],
                                     [x1, y1, z2], [x2, y1, z2], [x1, y2, z2],[x2, y2, z2]],dtype = nptype))
  #windows
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*1, y2, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*1, y2, z1+(z2-z1)/4*3]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*3, y2, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*3, y2, z1+(z2-z1)/4*3]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*1, y2, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*3, y2, z1+(z2-z1)/4*1]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*1, y2, z1+(z2-z1)/4*3],[x1+(x2-x1)/8*3, y2, z1+(z2-z1)/4*3]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*1, y2, z1+(z2-z1)/4*2],[x1+(x2-x1)/8*3, y2, z1+(z2-z1)/4*2]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*2, y2, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*2, y2, z1+(z2-z1)/4*3]],dtype = nptype))

  lines = np.append(lines, np.array([[x1+(x2-x1)/8*5, y2, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*5, y2, z1+(z2-z1)/4*3]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*7, y2, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*7, y2, z1+(z2-z1)/4*3]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*5, y2, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*7, y2, z1+(z2-z1)/4*1]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*5, y2, z1+(z2-z1)/4*3],[x1+(x2-x1)/8*7, y2, z1+(z2-z1)/4*3]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*5, y2, z1+(z2-z1)/4*2],[x1+(x2-x1)/8*7, y2, z1+(z2-z1)/4*2]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*6, y2, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*6, y2, z1+(z2-z1)/4*3]],dtype = nptype))

  lines = np.append(lines, np.array([[x1+(x2-x1)/8*1, y1, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*1, y1, z1+(z2-z1)/4*3]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*3, y1, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*3, y1, z1+(z2-z1)/4*3]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*1, y1, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*3, y1, z1+(z2-z1)/4*1]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*1, y1, z1+(z2-z1)/4*3],[x1+(x2-x1)/8*3, y1, z1+(z2-z1)/4*3]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*1, y1, z1+(z2-z1)/4*2],[x1+(x2-x1)/8*3, y1, z1+(z2-z1)/4*2]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*2, y1, z1+(z2-z1)/4*1],[x1+(x2-x1)/8*2, y1, z1+(z2-z1)/4*3]],dtype = nptype))

  #door
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*5, y1, z1+(z2-z1)/4*3],[x1+(x2-x1)/8*7, y1, z1+(z2-z1)/4*3]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*5, y1, 0.],[x1+(x2-x1)/8*5, y1, z1+(z2-z1)/4*3]],dtype = nptype))
  lines = np.append(lines, np.array([[x1+(x2-x1)/8*7, y1, 0.],[x1+(x2-x1)/8*7, y1, z1+(z2-z1)/4*3]],dtype = nptype))

  #ceiling
  #for i in range(int(y2-y1)-1):
  #  lines = np.append(lines, np.array([[x1,y1+(y2-y1)*(i+1)/int(y2-y1),z2],[x2,y1+(y2-y1)*(i+1)/int(y2-y1),z2]],dtype = nptype))
  #for i in range(int(x2-x1)-1):
  #  lines = np.append(lines, np.array([[x1+(x2-x1)*(i+1)/int(x2-x1),y1,z2],[x1+(x2-x1)*(i+1)/int(x2-x1),y2,z2]],dtype = nptype))

  #flour
  #for i in range(int(y2-y1)-1):
  #  lines = np.append(lines, np.array([[x1,y1+(y2-y1)*(i+1)/int(y2-y1),z1],[x2,y1+(y2-y1)*(i+1)/int(y2-y1),z1]],dtype = nptype))
  #for i in range(int(x2-x1)-1):
  #  lines = np.append(lines, np.array([[x1+(x2-x1)*(i+1)/int(x2-x1),y1,z1],[x1+(x2-x1)*(i+1)/int(x2-x1),y2,z1]],dtype = nptype))



#INFO if it is expected the 3D landmarks more densely, repeat the function more times
# draw the landmarks 
for i in range(28):
  landmarksBuilding(12, x1, x2, y1, y2, z1, z2)
#for i in range(18):
#  landmarksBuilding(10, x1, x2, y1, y2/2, z1, z2)
#for i in range(12):
#  landmarksBuilding(10, x1, x2, y1, y2, z1, z2/2)
#for i in range(9):
#  landmarksBuilding(10, x1, x2, y1, y2/4, z1, z2)

#manMadeLandmarks()

lines = np.reshape(lines,(-1,3))
stars = np.reshape(stars,(-1,3))
triangles = np.reshape(triangles,(-1,3))
cubes = np.reshape(cubes,(-1,3))

# save simulated landmarks to txt
linespath = os.path.abspath('..') + "/support_files/txt_data/" + "lines.txt"
starspath = os.path.abspath('..') + "/support_files/txt_data/" + "stars.txt"
trianglespath = os.path.abspath('..') + "/support_files/txt_data/" + "triangles.txt"
cubespath = os.path.abspath('..') + "/support_files/txt_data/" + "cubes.txt"

np.savetxt(linespath,lines)
np.savetxt(starspath,stars)
np.savetxt(trianglespath,triangles)
np.savetxt(cubespath,cubes)

#visualize
utils.visualizeLines3D(ax)
utils.visualizeStars3D(ax)
utils.visualizeTriangles3D(ax)
utils.visualizeCubes3D(ax)

utils.visualizeTrajactory3D(ax)

ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(x1, x2)
ax.set_ylim(y1, y2)
ax.set_zlim(z1, z2)
plt.show() 








