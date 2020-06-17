import os
import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv
import utils   
#import trajactoryGenerator

nptype = np.float64
IMAGENUM = 2000

savepath = os.path.abspath('..') + "/support_files/txt_data/" + "trajactory.txt"
data = np.loadtxt(savepath,dtype=nptype) 
data = np.reshape(data,(-1,8))

linespath = os.path.abspath('..') + "/support_files/txt_data/" + "lines.txt"
starspath = os.path.abspath('..') + "/support_files/txt_data/" + "stars.txt"
trianglespath = os.path.abspath('..') + "/support_files/txt_data/" + "triangles.txt"
cubespath = os.path.abspath('..') + "/support_files/txt_data/" + "cubes.txt"

lines = np.loadtxt(linespath,dtype=nptype)
stars = np.loadtxt(starspath,dtype=nptype)
triangles = np.loadtxt(trianglespath,dtype=nptype)
cubes = np.loadtxt(cubespath,dtype=nptype)

lines = np.reshape(lines,(-1,3))
stars = np.reshape(stars,(-1,3))
triangles = np.reshape(triangles,(-1,3))
cubes = np.reshape(cubes,(-1,3))

for i in range(IMAGENUM):#trajactoryGenerator.num):
  print('************this is the round: %d*************'%i)
  #transform c->W
  Rw = utils.q2R(data[i,4:8])
  tw = np.array([data[i,1],data[i,2],data[i,3]],dtype=nptype)
  #transform w->c
  Rc = (Rw.T).copy()
  tc = ((-np.dot(Rc,tw.T)).T).copy()

  #get the coordinates in camera frame
  lines_temp = lines.copy()
  stars_temp = stars.copy()
  triangles_temp = triangles.copy()
  cubes_temp = cubes.copy()

  lines_Pc = np.array([],dtype=nptype)
  stars_Pc = np.array([],dtype=nptype)
  triangles_Pc = np.array([],dtype=nptype)
  cubes_Pc = np.array([],dtype=nptype)

  lines_Pc = utils.Pw2Pc(lines_temp, Rc, tc)
  stars_Pc = utils.Pw2Pc(stars_temp, Rc, tc)
  triangles_Pc = utils.Pw2Pc(triangles_temp, Rc, tc)
  cubes_Pc = utils.Pw2Pc(cubes_temp, Rc, tc)

  #get the coordinates in normalized frame
  lines_Pn = np.array([],dtype=nptype)
  stars_Pn = np.array([],dtype=nptype)
  triangles_Pn = np.array([],dtype=nptype)
  cubes_Pn = np.array([],dtype=nptype)

  lines_Pn = utils.Pc2Pn(lines_Pc)
  stars_Pn = utils.Pc2Pn(stars_Pc)
  triangles_Pn = utils.Pc2Pn(triangles_Pc)
  cubes_Pn = utils.Pc2Pn(cubes_Pc)

  #generate the image background
  img = np.zeros((utils.image_height, utils.image_width), dtype=np.uint8)
  #img = utils.generate_background((utils.image_height, utils.image_width),100,0.01,0.05,50,300)
  img[img==0] = 255
  #get the coordinates in image frame
  lines_Pi = np.array([],dtype=nptype)
  stars_Pi = np.array([],dtype=nptype)
  triangles_Pi = np.array([],dtype=nptype)
  cubes_Pi = np.array([],dtype=nptype)

  lines_Pi = utils.Pn2Pi(img, lines_Pn)
  stars_Pi = utils.Pn2Pi(img, stars_Pn)
  triangles_Pi = utils.Pn2Pi(img, triangles_Pn)
  cubes_Pi = utils.Pn2Pi(img, cubes_Pn)

  #draw the camera views
  img = utils.visualizeLines2D(img, lines_Pi)
  img = utils.visualizeStars2D(img, stars_Pi)
  img = utils.visualizeTriangles2D(img, triangles_Pi)
  img = utils.visualizeCubes2D(img, cubes_Pi)
  #img = utils.add_salt_and_pepper(img)
  #img = utils.final_blur(img, kernel_size=(5, 5))

  #save data
  str =  os.path.abspath('..') + "/support_files/series_data/images/pure/" + "%d"%(i+1) + ".png"
  cv.imwrite(str,img)

  #img = utils.final_blur(img, kernel_size=(5, 5))
  #str =  os.path.abspath('..') + "/support_files/series_data/images/medium/" + "%d"%(i+1) + ".png"
  #cv.imwrite(str,img)

  lines_str =  os.path.abspath('..') + "/support_files/series_data/landmarksGroundtruth/lines/" + "%d"%(i+1) + ".txt"
  stars_str =  os.path.abspath('..') + "/support_files/series_data/landmarksGroundtruth/stars/" + "%d"%(i+1) + ".txt"
  triangles_str =  os.path.abspath('..') + "/support_files/series_data/landmarksGroundtruth/triangles/" + "%d"%(i+1) + ".txt"
  cubes_str =  os.path.abspath('..') + "/support_files/series_data/landmarksGroundtruth/cubes/" + "%d"%(i+1) + ".txt"

  print("lines_Pi: %d"%lines_Pi.shape[0])
  print("stars_Pi: %d"%stars_Pi.shape[0])
  print("triangles_Pi: %d"%triangles_Pi.shape[0])
  print("cubes_Pi: %d"%cubes_Pi.shape[0])

  np.savetxt(lines_str,lines_Pi)
  np.savetxt(stars_str,stars_Pi)
  np.savetxt(triangles_str,triangles_Pi)
  np.savetxt(cubes_str,cubes_Pi)

    
