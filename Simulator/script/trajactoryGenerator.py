# This is a trajactory generator from raw data EuRoC
# because raw data is 200Hz, so this is down sampling
# into 20Hz to simulate camera frequency
# warming: the input txt format must follow TUM style!

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils         

nptype = np.float64

#frequency of camera
freq = 20
#when(s) the selected data start from the raw data
start =0
#How long(s) that is expected to simulate(should not longer than raw data)
dura = 100
#how many data are selected
num = int(dura * freq)
#sample rate 
step = 200 / freq 
#to visually check the sample data with the raw data
flag_compare = 1
#whether to treat the first pose as world coordinate system
flag_change = 1
#whether to visually check the trajectory
flag_show = 0
#input the raw data that is selected
filepath = os.path.abspath('..') + "/support_files/EuRoC_traj_data/" + "V1_03_difficult.tum"
data_raw = np.loadtxt(filepath,dtype=nptype)

#the chosen data
data_select = np.array([],dtype = nptype)
for i in range(num):
   data_select = np.append(data_select, data_raw[start*200+i*step])
data_select = np.reshape(data_select, (-1,8))
print('sample number:%d'%data_select.shape[0])

#visualize
if (flag_compare==1):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot(data_raw[:,1],data_raw[:,2],data_raw[:,3],'b.',markersize=1, label='raw')
  ax.plot(data_select[:,1],data_select[:,2],data_select[:,3],'r.',markersize=4, label='choose')
  ax.legend()
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show()

#make the first pose of data_select to be the origin in world coordinate system
if (flag_change==1):
  #record the first data 
  x0 = data_select[0,1]
  y0 = data_select[0,2]
  z0 = data_select[0,3]
  qc0w = utils.quaternion_inverse(data_select[0,4:8])

  #make the first pose as origin
  data_select[0,1:8] = np.array([0., 0., 0., 0., 0., 0., 1.],dtype = nptype)

  #transform all the poses into new coordinate system
  for i in range(data_select.shape[0]-1):
    data_select[i+1,1] = data_select[i+1,1] - x0
    data_select[i+1,2] = data_select[i+1,2] - y0
    data_select[i+1,3] = data_select[i+1,3] - z0 
    data_select[i+1,4:8] = utils.quaternion_multiply(qc0w, data_select[i+1,4:8])

#visually check the trajectory
if (flag_show==1):
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot(data_select[:,1],data_select[:,2],data_select[:,3],'r.',markersize=4, label='trajactory')
  #ax.plot(data_select[:,1],data_select[:,2],data_select[:,3])
  ax.legend()
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.show()  

#output to txt file to record the simulated trajactory 
savepath = os.path.abspath('..') + "/support_files/txt_data/" + "trajactory.txt"
np.savetxt(savepath,data_select)
print('sample number:%d'%data_select.shape[0])



