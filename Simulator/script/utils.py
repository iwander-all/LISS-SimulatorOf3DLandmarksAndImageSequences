import os
import numpy as np
import random
import math as m
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
 

#Warming: the input data should follow TUM format!
nptype = np.float64
random_state = np.random.RandomState(None)
image_width = 1280
image_height = 960
flag_distorted = 1

#distortion_parameters:
k1 = -2.9545645106987750e-01
k2 = 8.6623215640186171e-02
p1 = 2.0132892276082517e-06 
p2 = 1.3924531371276508e-05

#projection_parameters:
fx = 4.6115862106007575e+02
fy = 4.5975286598073296e+02
cx = 3.6265929181685937e+02
cy = 2.4852105668448124e+02

def q2R(quaternion):
  #input:  quaternion
  #output: rotation matrix
  x, y, z, w = quaternion
  return np.array([[w*w+x*x-y*y-z*z, 2*(x*y-w*z),     2*(x*z+w*y)], 
                   [2*(x*y+w*z),     w*w-x*x+y*y-z*z, 2*(y*z-w*x)], 
                   [2*(x*z-w*y),     2*(y*z+w*x),     w*w-x*x-y*y+z*z]], dtype=nptype)

def quaternion_multiply(quaternion1, quaternion2): 
  #input:  2 quaternions
  #output: their product
  #q1 * q2 = (a1 + b1i + c1j + d1k) (a2 + b2i + c2j + d2k)
  b1, c1, d1, a1 = quaternion1 
  b2, c2, d2, a2 = quaternion2 
  return np.array([ a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2, 
                    a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, 
                    a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2, 
                    a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2], dtype=nptype)

def quaternion_inverse(quaternion): 
  #input:  quaternion
  #output: inverse of quaternion
  b, c, d, a = quaternion
  conjugation = np.array([-b, -c, -d, a], dtype=nptype)
  length2 = a*a + b*b + c*c + d*d
  return conjugation / length2
    
def scale(scaler1, scaler2, point):
  #input:  range of scalers and n*3 transpose of points
  #output: n*3 transpose of points
  point = np.reshape(point,(-1,3))
  points = point.copy()
  for i in range(points.shape[0]):
    rand_scaler = random.uniform(scaler1, scaler2)
    points[i] = rand_scaler * points[i]
  return points

def scaleCubes(scaler1, scaler2, point):
  #input:  range of scalers and n*3 transpose of points
  #output: n*3 transpose of points
  point = np.reshape(point,(-1,3))
  points = point.copy()
  rand_scaler = random.uniform(scaler1, scaler2)
  for i in range(points.shape[0]):
    points[i] = rand_scaler * points[i]
  return points

def affine(scaler1, scaler2, point):
  #input:  range of scalers and n*3 transpose of points
  #output: n*3 transpose of points
  point = np.reshape(point,(-1,3))
  points = point.copy()
  rand11 = random.uniform(scaler1, scaler2)
  rand22 = random.uniform(scaler1, scaler2)
  rand33 = random.uniform(scaler1, scaler2)
  rand12 = random.uniform(-0.1, 0.1)
  rand13 = random.uniform(-0.1, 0.1)
  rand21 = random.uniform(-0.1, 0.1)
  rand23 = random.uniform(-0.1, 0.1)
  rand31 = random.uniform(-0.1, 0.1)
  rand32 = random.uniform(-0.1, 0.1)
  A = np.array([[rand11, rand12, rand13],
                [rand21, rand22, rand23],
                [rand31, rand32, rand33]], dtype=nptype)
  for i in range(points.shape[0]):
    points[i] = np.dot(A, points[i])
  return points

def EulerRotate(a1, a2, a3):
  #input:  3 euler angles
  #output: 3*3 rotation matrix
  return np.array([[m.cos(a1)*m.cos(a2),m.cos(a1)*m.sin(a2)*m.sin(a3)-m.sin(a1)*m.cos(a3),m.cos(a1)*m.sin(a2)*m.cos(a3)+m.sin(a1)*m.sin(a3)],
                   [m.sin(a1)*m.cos(a2),m.sin(a1)*m.sin(a2)*m.sin(a3)+m.cos(a1)*m.cos(a3),m.sin(a1)*m.sin(a2)*m.cos(a3)-m.cos(a1)*m.sin(a3)],
                   [-m.sin(a2),m.cos(a2)*m.sin(a3),m.cos(a2)*m.cos(a3)]],dtype=nptype)

def multi(R, points):
  #input:  3*3 rotation matrix, n*3 transpose of points (in favor of txt and array saving)
  #output: n*3 transpose of points  
  points = np.reshape(points,(-1,3))
  return (np.dot(R,points.T)).T

def translate(t1, t2, t3, points):
  #input:  3*1 translation vector, n*3 transpose of points (in favor of txt and array saving)
  #output: n*3 transpose of points 
  points = np.reshape(points,(-1,3)) 
  for i in range(points.shape[0]):
    points[i,0] = points[i,0] + t1 
    points[i,1] = points[i,1] + t2
    points[i,2] = points[i,2] + t3
  return points

def visualizeLines3D(ax):
  #input:  ax viewer, n*3 transpose of points
  #output: visualize
  linespath = os.path.abspath('..') + "/support_files/txt_data/" + "lines.txt"
  lines = np.loadtxt(linespath,dtype=nptype)
  lines = np.reshape(lines,(-1,3))
  for i in range(lines.shape[0]/2):  
    x = [lines[2*i,0], lines[2*i+1,0]]
    y = [lines[2*i,1], lines[2*i+1,1]]
    z = [lines[2*i,2], lines[2*i+1,2]]
    ax.plot(x,y,z,'-',markersize=1)  
  
def visualizeStars3D(ax):
  #input:  ax viewer, n*3 transpose of points
  #output: visualize
  starspath = os.path.abspath('..') + "/support_files/txt_data/" + "stars.txt"
  stars = np.loadtxt(starspath,dtype=nptype)
  stars = np.reshape(stars,(-1,3))
  for i in range(stars.shape[0]/7):
    for j in range(6):
      x = [stars[7*i,0], stars[7*i+j+1,0]]
      y = [stars[7*i,1], stars[7*i+j+1,1]]
      z = [stars[7*i,2], stars[7*i+j+1,2]]
      ax.plot(x, y, z,'-',markersize=1)

def visualizeTriangles3D(ax):
  #input:  ax viewer, n*3 transpose of points
  #output: visualize
  trianglespath = os.path.abspath('..') + "/support_files/txt_data/" + "triangles.txt"
  triangles = np.loadtxt(trianglespath,dtype=nptype)
  triangles = np.reshape(triangles,(-1,3))
  for i in range(triangles.shape[0]/3):
    for j in range(3):
      x = [triangles[3*i+j,0], triangles[3*i+(j+1)%3,0]]
      y = [triangles[3*i+j,1], triangles[3*i+(j+1)%3,1]]
      z = [triangles[3*i+j,2], triangles[3*i+(j+1)%3,2]]
      ax.plot(x, y, z,'-',markersize=1)

def visualizeCubes3D(ax):
  #input:  ax viewer, n*3 transpose of points
  #output: visualize
  cubespath = os.path.abspath('..') + "/support_files/txt_data/" + "cubes.txt"
  cubes = np.loadtxt(cubespath,dtype=nptype)
  cubes = np.reshape(cubes,(-1,3))
  for i in range(cubes.shape[0]/8):
    for j in range(2):
      for k in range(2):
        x = [cubes[8*i+4*j+k+1,0],cubes[8*i+4*j+3*k,0]]
        y = [cubes[8*i+4*j+k+1,1],cubes[8*i+4*j+3*k,1]]
        z = [cubes[8*i+4*j+k+1,2],cubes[8*i+4*j+3*k,2]]
        ax.plot(x, y, z,'-',markersize=1)
      for k in range(2):
        x = [cubes[8*i+4*j+k+1,0],cubes[8*i+4*j+3*(1-k),0]]
        y = [cubes[8*i+4*j+k+1,1],cubes[8*i+4*j+3*(1-k),1]]
        z = [cubes[8*i+4*j+k+1,2],cubes[8*i+4*j+3*(1-k),2]]
        ax.plot(x, y, z,'-',markersize=1)
    for j in range(4):
      x0 = [cubes[8*i+j,0],cubes[8*i+4+j,0]]
      y0 = [cubes[8*i+j,1],cubes[8*i+4+j,1]]
      z0 = [cubes[8*i+j,2],cubes[8*i+4+j,2]]
      ax.plot(x0, y0, z0,'-',markersize=1)
  
def visualizeTrajactory3D(ax):
  #input:  ax viewer
  #output: visualize
  savepath = os.path.abspath('..') + "/support_files/txt_data/" + "trajactory.txt"
  #savepath = os.path.abspath('..') + "/support_files/txt_data/" + "V1_03_difficult_150_400_trajectory.txt"
  #savepath = os.path.abspath('..') + "/support_files/txt_data/" + "MH_03_medium_2000_2100_trajectory.txt"
  #savepath = os.path.abspath('..') + "/support_files/txt_data/" + "FAST_PnP_trajectory.txt"
  #savepath = os.path.abspath('..') + "/support_files/txt_data/" + "SIFT_PnP_trajectory.txt"
  data = np.loadtxt(savepath,dtype=nptype) 
  data = np.reshape(data,(-1,8)) 
  ax.plot(data[:,1],data[:,2],data[:,3],'r.',markersize=2)

def Pw2Pc(point, Rc, tc):
  #input:  n*3 transpose of points in world frame
  #        transform from world to camera
  #output: n*3 transpose of points in camera frame
  points = point.copy()
  for i in range(points.shape[0]):
    points[i] = (np.dot(Rc, points[i].T) + tc.T).T
    if (points[i,2] <= 0):
      points[i] = np.array([-1., -1., -1.],dtype=nptype)
    #print(points[i])
  return points

def Pc2Pn(point):
  #input:  n*3 transpose of points in camera frame
  #output: n*3 transpose of points in normalized frame  
  points = point.copy()
  for i in range(points.shape[0]):
    if (points[i,0] == -1.):
      continue
    else:
      points[i] = np.array([points[i,0]/points[i,2], points[i,1]/points[i,2], 1.],dtype=nptype)
      if (flag_distorted == 1):
        r = m.sqrt(points[i,0]*points[i,0] + points[i,1]*points[i,1])
        points[i,0] = points[i,0] * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * points[i,0] * points[i,1] + p2 * (r * r + 2 * points[i,0] * points[i,0])
        points[i,1] = points[i,1] * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * points[i,1] * points[i,1]) + 2 * p2 * points[i,0] * points[i,1]
    #print(points[i])
  return points 

def Pn2Pi(img, point):
  #input:  n*3 transpose of points in normalized frame
  #output: n*3 transpose of points in pixel frame
  points = point.copy()
  K = np.array([[fx, 0., cx],
                [0., fy, cy],
                [0., 0., 1.]],dtype=nptype)
  for i in range(points.shape[0]):
    if (points[i,0] == -1.):
      continue
    else:
      points[i] = (np.dot(K, points[i].T)).T
  points = np.reshape(points, (-1, 3))
  for i in range(points.shape[0]):
    if ((points[i,0] > img.shape[0] or points[i,0] < 0) and
        (points[i,1] > img.shape[1] or points[i,1] < 0)):
      points[i] = np.array([-1., -1., -1.],dtype=nptype)
    #print(points[i])
  points = np.reshape(points, (-1, 3))
  return points

def get_random_color(background_color):
    """ Output a random scalar in grayscale with a least a small
        contrast with the background color """
    color = random.randint(0, 256)
    if abs(color - background_color) < 30:  # not enough contrast
        color = (color + 128) % 256
    return color

def generate_background(size=(960, 1280), nb_blobs=100, min_rad_ratio=0.01,
                        max_rad_ratio=0.05, min_kernel_size=50, max_kernel_size=300):
    """ Generate a customized background image
    Parameters:
      size: size of the image
      nb_blobs: number of circles to draw
      min_rad_ratio: the radius of blobs is at least min_rad_size * max(size)
      max_rad_ratio: the radius of blobs is at most max_rad_size * max(size)
      min_kernel_size: minimal size of the kernel
      max_kernel_size: maximal size of the kernel
    """
    img = np.zeros(size, dtype=np.uint8)
    #img[img==0] = 255
    dim = max(size)
    cv.randu(img, 100, 255)
    cv.threshold(img, random_state.randint(100,255), 255, cv.THRESH_BINARY, img)
    background_color = int(np.mean(img))
    blobs = np.concatenate([random_state.randint(0, size[1], size=(nb_blobs, 1)),
                            random_state.randint(0, size[0], size=(nb_blobs, 1))],
                           axis=1)
    for i in range(nb_blobs):
        col = get_random_color(background_color)
        cv.circle(img, (blobs[i][0], blobs[i][1]),
                  np.random.randint(int(dim * min_rad_ratio),
                                    int(dim * max_rad_ratio)),
                  col, -1)
    kernel_size = random_state.randint(min_kernel_size, max_kernel_size)
    cv.blur(img, (kernel_size, kernel_size), img)
    return img

def visualizeLines2D(img, lines_Pi):
  #background_color = int(np.mean(img))
  #col = get_random_color(background_color)
  col = 100
  thickness = 2
  for i in range(lines_Pi.shape[0]/2):
    if (lines_Pi[2*i,0] > 0 and lines_Pi[2*i+1,0] > 0):  
       cv.line(img, (int(lines_Pi[2*i,0]), int(lines_Pi[2*i,1])), (int(lines_Pi[2*i+1,0]), int(lines_Pi[2*i+1,1])), col, thickness)
  return img

def visualizeStars2D(img, stars_Pi):
  #background_color = int(np.mean(img))
  #col = get_random_color(background_color)
  col = 100
  thickness = 2
  for i in range(stars_Pi.shape[0]/7):
    for j in range(6):
      if (stars_Pi[7*i,0] > 0 and stars_Pi[7*i+j+1,0] > 0):
        cv.line(img, (int(stars_Pi[7*i,0]), int(stars_Pi[7*i,1])), (int(stars_Pi[7*i+j+1,0]), int(stars_Pi[7*i+j+1,1])), col, thickness)
  return img

def visualizeTriangles2D(img, triangles_Pi):
  #background_color = int(np.mean(img))
  #col = get_random_color(background_color)
  col = 100
  thickness = 2
  for i in range(triangles_Pi.shape[0]/3):
    for j in range(3):
      if (triangles_Pi[3*i+j,0] > 0 and triangles_Pi[3*i+(j+1)%3,0] > 0):
        cv.line(img, (int(triangles_Pi[3*i+j,0]), int(triangles_Pi[3*i+j,1])), 
                     (int(triangles_Pi[3*i+(j+1)%3,0]), int(triangles_Pi[3*i+(j+1)%3,1])), col, thickness)
  return img

def visualizeCubes2D(img, cubes_Pi):
  #background_color = int(np.mean(img))
  #col = get_random_color(background_color)
  col = 100
  thickness = 2
  cubes_Pi = np.reshape(cubes_Pi, (-1,3))
  for i in range(cubes_Pi.shape[0]/8):
    for j in range(2):
      for k in range(2):
        if (cubes_Pi[8*i+4*j+k+1,0] > 0 and cubes_Pi[8*i+4*j+3*k,0] > 0):
          cv.line(img, (int(cubes_Pi[8*i+4*j+k+1,0]), int(cubes_Pi[8*i+4*j+k+1,1])), 
                       (int(cubes_Pi[8*i+4*j+3*k,0]), int(cubes_Pi[8*i+4*j+3*k,1])), col, thickness)
      for k in range(2):
        if (cubes_Pi[8*i+4*j+k+1,0] > 0 and cubes_Pi[8*i+4*j+3*(1-k),0] > 0):
          cv.line(img, (int(cubes_Pi[8*i+4*j+k+1,0]), int(cubes_Pi[8*i+4*j+k+1,1])), 
                       (int(cubes_Pi[8*i+4*j+3*(1-k),0]), int(cubes_Pi[8*i+4*j+3*(1-k),1])), col, thickness)
    for j in range(4):
      if (cubes_Pi[8*i+j,0] > 0 and cubes_Pi[8*i+4+j,0] > 0):
        cv.line(img, (int(cubes_Pi[8*i+j,0]), int(cubes_Pi[8*i+j,1])), 
                     (int(cubes_Pi[8*i+4+j,0]), int(cubes_Pi[8*i+4+j,1])), col, thickness)
  return img

def add_salt_and_pepper(img):
    """ Add salt and pepper noise to an image """
    noise = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    cv.randu(noise, 0, 255)
    black = noise < 30
    white = noise > 225
    img[white > 0] = 255
    img[black > 0] = 0
    cv.blur(img, (5, 5), img)
    #return np.empty((0, 2), dtype=np.int)
    return img

def final_blur(img, kernel_size=(5, 5)):
    """ Apply a final Gaussian blur to the image
    Parameters:
      kernel_size: size of the kernel
    """
    cv.GaussianBlur(img, kernel_size, 0, img)
    return img
