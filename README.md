# LISS
## Simulator Of 3D Landmarks And Image Sequences Based On Line Features

**17 Jun 2020**:  This is the  **1st** Edition, for Chinese version please click [Introduction](https://blog.csdn.net/iwanderu/article/details/106812369).

This is a generator tool used for 3D landmarks building based on line features which can be used to test VO/SLAM system. It can be easily changed into VIO-SLAM pattern.


**Authors:** [iwander。](https://blog.csdn.net/iwanderu/article/details/106812369)

**Videos:** There is a introduction video to show how to use this tool.

<a href="https://blog.csdn.net/iwanderu/article/details/106812369" target="_blank"><img src="http://img.youtube.com/vi/mv_9snb_bKs/0.jpg" 
alt="euroc" width="240" height="180" border="10" /></a>

## 1. Prerequisites
 **python** 
```
python 2 or python 3.
numpy.
pyplot.
openCV 3.
```

## 2. Build LISS on PC
Clone the repository and run:
```
    cd ~/home/
    git clone https://github.com/iwander-all/LISS-SimulatorOf3DLandmarksAndImageSequences.git
    cd script
```

## 3. Run with your device 
3.1 Visually check the trajectories
```
    python traj_visual.py
```
<a><img src="http://img.youtube.com/vi/mv_9snb_bKs/0.jpg" alt="V1_03_difficult" width="240" height="180" border="10" /></a>


3.2 Build trajectory
```
    python trajactoryGenerator.py
```
The red line is selected.
<a><img src="http://img.youtube.com/vi/mv_9snb_bKs/0.jpg" alt="V1_03_difficult" width="240" height="180" border="10" /></a>

3.3 Build 3D landmarks
```
    python lineGenerator.py
```
The landmarks are randomly built, so if you want to check the built one, please use:
```
    python visualize.py
```
<a><img src="http://img.youtube.com/vi/mv_9snb_bKs/0.jpg" alt="V1_03_difficult" width="240" height="180" border="10" /></a>

3.4 Build images
```
    python imageGenerator.py
```
<a><img src="http://img.youtube.com/vi/mv_9snb_bKs/0.jpg" alt="201" width="240" height="180" border="10" /></a>
<a><img src="http://img.youtube.com/vi/mv_9snb_bKs/0.jpg" alt="202" width="240" height="180" border="10" /></a>
<a><img src="http://img.youtube.com/vi/mv_9snb_bKs/0.jpg" alt="203" width="240" height="180" border="10" /></a>

## 4. Acknowledgements
I use [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) for trajectory and poses and the parameters for camera model.

## 5. Licence
The source code is released under [Apache License 2.0](http://www.apache.org/licenses/) license.

## 6. Further Work
I am still working on improving the code reliability and further work for VIO-SLAM pattern is expected. For any technical issues, please contact me <https://blog.csdn.net/iwanderu/> .

