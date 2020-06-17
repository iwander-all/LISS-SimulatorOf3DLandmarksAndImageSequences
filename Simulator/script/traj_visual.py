# This is a visualizer of trajactory, and the resource data is
# in the path ${Simulator}/support_files/EuRoC_data/XX_0X_XX.txt.
# All the data is in the format of tum, which is (200Hz)
#
# time, x, y, z, qx, qy, qz, qw, here is the boundary records:
#                       x             y            z             number
# MH_01_easy        (-3~ 5)=8     (-4~10)=14  (-1.5~1.5)=3       36382
# MH_02_easy        (-3~ 5)=8     (-4~10)=14  (-1.5~1.5)=3       29993
# MH_03_medium      (-2~14)=16    (-4~ 8)=12  (-1  ~1.5)=2.5     26302
# MH_04_difficult   (-5~20)=25    (-6~12)=18  (0.5 ~  4)=3.5     19753
# MH_05_difficult   (-5~20)=25    (-6~12)=18  (0   ~  4)=4       22212
# V1_01_easy        (-3~ 3)=6     (-3~ 4)=7   (0.8~   2)=1.2     28712
# V1_02_medium      (-2.5~2)=4.5  (-2~ 4)=6   (0.8~ 2.2)=1.4     16702
# V1_03_difficult   (-2.5~2)=4.5  (-2~ 4)=6   (0.8~ 2.6)=1.8     20932
# V2_01_easy        (-4~  3)=7    (-2~ 4)=6   (0.8~ 2.2)=1.4     22401
# V2_02_medium      (-4~  3)=7    (-3~ 4)=7   (0.5 ~  3)=2.5     23091
# V2_03_difficult   (-4~  2)=6    (-2~ 4)=6   (0.5 ~  3)=2.5     22970

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

nptype = np.float64
filepath = os.path.abspath('..') + "/support_files/EuRoC_traj_data/" + "V1_03_difficult.tum"

data_raw = np.loadtxt(filepath,dtype=nptype)
print("how many data:%d"%data_raw.shape[0])


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(data_raw[:,1],data_raw[:,2],data_raw[:,3],'-',markersize=1, label='control line')
ax.plot(data_raw[:,1],data_raw[:,2],data_raw[:,3],'r.',markersize=1, label='IMU')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
