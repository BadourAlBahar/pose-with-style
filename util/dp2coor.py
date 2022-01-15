import os
from PIL import Image
import numpy as np
from scipy.interpolate import griddata
import cv2
import argparse

def getSymXYcoordinates(iuv, resolution = 256):
    xy, xyMask = getXYcoor(iuv, resolution = resolution)
    f_xy, f_xyMask = getXYcoor(flip_iuv(np.copy(iuv)), resolution = resolution)
    f_xyMask = np.clip(f_xyMask-xyMask, a_min=0, a_max=1)
    # combine actual + symmetric
    combined_texture = xy*np.expand_dims(xyMask,2) + f_xy*np.expand_dims(f_xyMask,2)
    combined_mask = np.clip(xyMask+f_xyMask, a_min=0, a_max=1)
    return combined_texture, combined_mask, f_xyMask

def flip_iuv(iuv):
    POINT_LABEL_SYMMETRIES = [ 0, 1, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23]
    i = iuv[:,:,0]
    u = iuv[:,:,1]
    v = iuv[:,:,2]
    i_old = np.copy(i)
    for part in range(24):
        if (part + 1) in i_old:
            annot_indices_i = i_old == (part + 1)
            if POINT_LABEL_SYMMETRIES[part + 1] != part + 1:
                    i[annot_indices_i] = POINT_LABEL_SYMMETRIES[part + 1]
            if part == 22 or part == 23 or part == 2 or part == 3 : #head and hands
                    u[annot_indices_i] = 255-u[annot_indices_i]
            if part == 0 or part == 1: # torso
                    v[annot_indices_i] = 255-v[annot_indices_i]
    return np.stack([i,u,v],2)

def getXYcoor(iuv, resolution = 256):
    x, y, u, v = mapper(iuv, resolution)
    # A meshgrid of pixel coordinates
    nx, ny = resolution, resolution
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny, 1))
    ## get x,y coordinates
    uv_y = griddata((v, u), y, (Y, X), method='linear')
    uv_y_ = griddata((v, u), y, (Y, X), method='nearest')
    uv_y[np.isnan(uv_y)] = uv_y_[np.isnan(uv_y)]
    uv_x = griddata((v, u), x, (Y, X), method='linear')
    uv_x_ = griddata((v, u), x, (Y, X), method='nearest')
    uv_x[np.isnan(uv_x)] = uv_x_[np.isnan(uv_x)]
    # get mask
    uv_mask = np.zeros((ny,nx))
    uv_mask[np.ceil(v).astype(int),np.ceil(u).astype(int)]=1
    uv_mask[np.floor(v).astype(int),np.floor(u).astype(int)]=1
    uv_mask[np.ceil(v).astype(int),np.floor(u).astype(int)]=1
    uv_mask[np.floor(v).astype(int),np.ceil(u).astype(int)]=1
    kernel = np.ones((3,3),np.uint8)
    uv_mask_d = cv2.dilate(uv_mask,kernel,iterations = 1)
    # update
    coor_x = uv_x * uv_mask_d
    coor_y = uv_y * uv_mask_d
    coor_xy = np.stack([coor_x, coor_y], 2)
    return coor_xy, uv_mask_d

def mapper(iuv, resolution=256):
    dp_uv_lookup_256_np = np.load('util/dp_uv_lookup_256.npy')
    H, W, _ = iuv.shape
    iuv_raw = iuv[iuv[:, :, 0] > 0]
    x = np.linspace(0, W-1, W).astype(np.int)
    y = np.linspace(0, H-1, H).astype(np.int)
    xx, yy = np.meshgrid(x, y)
    xx_rgb = xx[iuv[:, :, 0] > 0]
    yy_rgb = yy[iuv[:, :, 0] > 0]
    # modify i to start from 0... 0-23
    i = iuv_raw[:, 0] - 1
    u = iuv_raw[:, 1]
    v = iuv_raw[:, 2]
    uv_smpl = dp_uv_lookup_256_np[
    i.astype(np.int),
    v.astype(np.int),
    u.astype(np.int)
    ]
    u_f = uv_smpl[:, 0] * (resolution - 1)
    v_f = (1 - uv_smpl[:, 1]) * (resolution - 1)
    return xx_rgb, yy_rgb, u_f, v_f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', type=str, help="path to image file to process. ex: ./train.lst")
    parser.add_argument("--save_path", type=str, help="path to save the uv data")
    parser.add_argument("--dp_path", type=str, help="path to densepose data")
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    images = []
    f = open(args.image_file, 'r')
    for lines in f:
        lines = lines.strip()
        images.append(lines)

    for i in range(len(images)):
        im_name = images[i]
        print ('%d/%d'%(i+1, len(images)))

        dp = os.path.join(args.dp_path, im_name.split('.')[0]+'_iuv.png')

        iuv = np.array(Image.open(dp))
        h, w, _ = iuv.shape
        if np.sum(iuv[:,:,0]==0)==(h*w):
            print ('no human: invalid image %d: %s'%(i, im_name))
        else:
            uv_coor, uv_mask, uv_symm_mask  = getSymXYcoordinates(iuv, resolution = 512)
            np.save(os.path.join(args.save_path, '%s_uv_coor.npy'%(im_name.split('.')[0])), uv_coor)
            mask_im = Image.fromarray((uv_mask*255).astype(np.uint8))
            mask_im.save(os.path.join(args.save_path, im_name.split('.')[0]+'_uv_mask.png'))
            mask_im = Image.fromarray((uv_symm_mask*255).astype(np.uint8))
            mask_im.save(os.path.join(args.save_path, im_name.split('.')[0]+'_uv_symm_mask.png'))
