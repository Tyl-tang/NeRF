import os
import json
import torch
import numpy as np
#from PIL import Image
#import matplotlib.pyplot as plt
import imageio.v2 as imageio

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


data_path =r'E:\Deeplearning\NeRF\Datasets\lego'
def load_blenderdata(data_path,testskip=8):
    split =['train','val','test']
    metas = {}
    for s in split:
        with open(os.path.join(data_path,f'transforms_{s}.json',),'r') as f:
            metas[s] = json.load(f)


    all_imgs = []
    all_poses = []
    counts =[0]
    camera_angle =[]

    for s in split:
        meta = metas[s]
        camera_angle = float(meta['camera_angle_x'])
        imgs = []
        poses =[]
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
        for frame in meta['frames'][::skip]:
            fname = os.path.join(data_path,frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))  #imageio.imread读取图像，返回的是图像的像素值数组
            poses.append(np.array(frame['transform_matrix']))
        imgs =(np.array(imgs)/255).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(imgs.shape[0])
        all_imgs.append(imgs)#all_imgs是一个列表，里面存放的是3组imgs的像素值。list没有shape属性
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    imgs = np.concatenate(all_imgs, 0)#将3组imgs的像素值拼接在一起,形成np数组
    poses = np.concatenate(all_poses, 0)
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    #####根据相机内参矩阵计算相机的焦距

    H,W = imgs[0].shape[:2]###############注意imgs的各个轴表示的含义
    focal = 0.5*(H/(np.tan(camera_angle/2)))
    K = np.array([[focal,0,W/2],[0,focal,H/2],[0,0,1]])



    return imgs, poses, render_poses,[H, W, focal],i_split

