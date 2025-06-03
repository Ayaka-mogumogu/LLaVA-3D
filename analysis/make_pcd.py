import numpy as np
import open3d as o3d
import cv2
import torch
import torch.nn.functional as F

def create_pointcloud_from_rgbd(color_img, depth_img, intrinsic_matrix, extrinsic_matrix=None):
    """
    カラー画像と深度画像から点群を生成する関数
    
    Args:
        color_img (np.ndarray): カラー画像 (H, W, 3)
        depth_img (np.ndarray): 深度画像 (H, W)
        intrinsic_matrix (np.ndarray): カメラ内部パラメータ (3, 3)
        extrinsic_matrix (np.ndarray, optional): カメラ外部パラメータ (4, 4)
    
    Returns:
        o3d.geometry.PointCloud: 生成された点群
    """
    # 深度画像の正規化（必要に応じて）
    if depth_img.max() > 1.0:
        depth_img = depth_img / depth_img.max()
    
    # カラー画像と深度画像からRGBD画像を作成
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_img),
        o3d.geometry.Image(depth_img),
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )
    
    # カメラ内部パラメータからカメラパラメータを作成
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        color_img.shape[1],  # width
        color_img.shape[0],  # height
        intrinsic_matrix[0, 0],  # fx
        intrinsic_matrix[1, 1],  # fy
        intrinsic_matrix[0, 2],  # cx
        intrinsic_matrix[1, 2]   # cy
    )
    
    # RGBD画像から点群を生成
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )
    
    # カメラ外部パラメータが指定されている場合、点群を変換
    if extrinsic_matrix is not None:
        pcd.transform(extrinsic_matrix)
    
    return pcd