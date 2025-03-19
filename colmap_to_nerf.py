import os
import numpy as np

def read_cameras(file):
    with open(file,'r') as f:
        lines = f.readlines()
        cameras = {}
        for line in lines[3:] :
            elements = line.split()
            if len(elements) < 4:
                continue
            camera_id = int(elements[0])
            params = list(map(float,elements[4:]))
            cameras[camera_id] = params
    return cameras

def read_images(file):
    with open(file,'r') as f:
        lines = f.readlines()
        images = {}
        for line in lines[4:] :
            elements = line.split()
            if len(elements) < 4:
                continue
            image_id = int(elements[0])
            qw,qx,qy,qz = map(float,elements[1:5])
            tx,ty,tz = map(float,elements[5:8])
            path = elements[-1]
            images[image_id] = {
                "q":np.array([qw,qx,qy,qz]),
                "t":np.array([tx,ty,tz]),
                "path":path
            }
    return images


def read_points3D(file):
    with open(file,'r') as f:
        lines = f.readlines()
