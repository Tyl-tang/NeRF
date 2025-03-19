import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio.v2 as imageio


datadir = r"E:\Deeplearning\NeRF\nerf-pytorch-master\nerf-pytorch-master\data\llff\fern"


def _load_pose_data(datadir):

   #在加载图片的同时也缩放图像

    poses_arr = np.load(os.path.join(datadir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape(-1, 3, 5).transpose((1, 2, 0))
    bounds = poses_arr[:, -2:].transpose((1, 0))

    images_dir = os.path.join(datadir,'images_8')
    all_files = os.listdir(images_dir)
    images = [os.path.join(images_dir,f)for f in all_files if f.endswith('.png') or f.endswith('.jpg')]
    images_sorted = sorted(images)
    #first_file = images_sorted[0]   用第一张图片来测试
    #img0 = os.path.join(images_dir,images_sorted[0])
    #image_data = imageio.imread(img0)
    #height, width, channels = image_data.shape

    imgs =[imageio.imread(f)[..., :3]/255 for f in images_sorted]
    imgs = np.stack(imgs,-1)


    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return imgs, poses, bounds


def load_llff_data(datadir):
    imgs, poses, bounds = _load_pose_data(datadir)
    print('Loaded', datadir, bounds.min(), bounds.max())

    #将llff坐标系下的坐标转换到nerf坐标系下的坐标
    poses = np.concatenate([poses[:,1:2,:], -poses[:,0:1,:],poses[:,2:,:]],1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)#将（3，5，20）转换为（20，3，5）
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bounds = np.moveaxis(bounds, -1, 0).astype(np.float32)
    poses = recenter_poses(poses)
    return images,poses,bounds








def recenter_poses(poses):
    poses_ =poses.copy()
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)#c2w的形状为（3，4）（x,y,z,t,）,不能直接求逆
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)#将c2w的最后一行添加（0,0,0,1）,变为4×4矩阵
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])#bottom的形状为（20，1，4）
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)#将poses的最后一行添加（0,0,0,1）,变为(20,4,4)
    poses = np.linalg.inv(c2w) @ poses#np.linalg.inv(c2w)表示对c2w求逆，得到w2c的矩阵，代表的是从世界坐标系到相机坐标系的变换，这步操作将所有位姿聚集到相机中心，与相机中心朝向一致
    poses_ [:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses



def poses_avg(poses):
    hwf = poses[0, :3, -1:]#poses 的形状是（20，3，5），刚好对应[height, width, focal]，
    center = poses[:, :3, 3].mean(0)#取出20张照片相机的平移，形状为（20，3）mean(0)表示沿第0维求平均值，即求出20张照片位姿平移的平均值，的（1，3）
    vec2 = np.normalize(poses[:, :3, 2].sum(0) )#取出20张照片相机的z轴方向，形状为（20，3），再沿0维方向求和，normalize表示归一化，即求出20张照片相机z轴方向求和后的单位向量
    up = poses[:, :3,1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2,up,center)],1)#c2w的形状为（3，4）（x,y,z,t）
    #c2w = np.concatenate([viewmatrix(vec2,up,center)],hwf,1) c2w的形状为（3，5）（x,y,z,t,hwf）
    return c2w  #以多个相机的平均位姿作为c2w矩阵




def viewmatrix(vec, up, pos):
    vec2  = np.normalize(vec)
    vec1 =  up
    vec0 =np.normalize(np.cross(vec1, vec2))#X轴
    vec1 =np.normalie(np.cross(vec2, vec0))#X和Z叉乘得到Y轴
    m = np.stack([vec0, vec1, vec2, pos], 1)#  c2w是根据归一化的Z轴，与向上的Y轴的和，得到X轴
    return m