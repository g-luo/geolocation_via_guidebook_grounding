import os
import numpy as np
import math
from pathlib import Path
import scipy as sp
from PIL import Image
from scipy import misc
from scipy import ndimage
from scipy import linalg
from scipy import interpolate
import sys
import glob
from tqdm import tqdm
import json

def pano2cutout(img,
    yaw=0, pitch=0, hfov=90,
    cutout_type='perspective',
    outdims=(1024, 768)
    ):
    # default pitch was 4
    '''
    Input:  yaw (default 0), pitch (default 0) - in degrees (0,360), resp. (-180,180)
            hfov (default 90) - horizontal field of view in degrees
            cutout_type (default 'perspective') - perspective or affine (camera plane aligned
                                                  with vertical axis)
            outdims (default (1024, 768) )- tuple of int, dimension of output image
    '''

    iimh, iimw, _ = img.shape
    oimw, oimh = outdims

    hfov = hfov*math.pi/180.0
    yaw   = yaw*math.pi/180.0
    pitch = -pitch*math.pi/180.0


    X,Y,Z = _createImageGrid(oimh, oimw, hfov)
    # Switch: perspective vs. rectified cutout
    if cutout_type == 'rectified':
        Xt,Yt,Zt  = _shiftImageGrid(X, Y, Z, pitch)            # offset Y coordinate
    elif cutout_type == 'perspective':
        Xt,Yt,Zt  = _rotateImageGrid(X, Y, Z, pitch) # image grid with pitch
    else:
        raise RuntimeError('Unknow "cutout_type"! "rectified" or "perspective" is available')
    # Cartesian to spgerical coordinates (rays)
    Theta,Phi = _imageGrid2spherical(Xt, Yt, Zt)
    # Yaw offset: 0,pi:google car direction, +-pi/2:street side
    Theta = Theta + yaw
    # Sampling points
    U,V = _spherical2panoImage(Theta, Phi, iimw, iimh)
    # Interpolation
    aux = np.hstack([img[:,:,0], img[:,0:1,0]])
    r = ndimage.map_coordinates(aux, [V,U], prefilter=False)
    aux = np.hstack([img[:,:,1], img[:,0:1,1]])
    g = ndimage.map_coordinates(aux, [V,U], prefilter=False)
    aux = np.hstack([img[:,:,2], img[:,0:1,2]])
    b = ndimage.map_coordinates(aux, [V,U], prefilter=False)

    return np.dstack([r,g,b])

def save(fname_out, img_out):
    '''
    Saves current panorama cut
    '''
    img_out = Image.fromarray(img_out)
    img_out.save(fname_out)

def _createImageGrid(oimh, oimw, hfov):
    '''
    Creates a grid of image coordinates
    oimh,oimw: output image dimensions in pixesl
    hfov: horizontal field of view in radians (I hope so...)
    '''
    # print(hfov/2*180/math.pi)
    # print(oimw)

    f = (oimw-1)/(2*math.tan(hfov/2))       # focal length [pix]

    ouc = (oimw)/2.0 - 0.5                      # output image center u,v
    ovc = (oimh)/2.0 - 0.5
    #iuc = (iimw+1)/2; ivc=(iimh+1)/2;          # input image center
    # Tangent plane to unit sphere mapping
    X, Y = np.meshgrid(range(oimw), range(oimh))
    X, Y = X.astype('float'), Y.astype('float')
    X -= ouc
    Y -= ovc                    # shift origin to the image center
    Z = f + np.zeros(X.shape)
    return X, Y, Z

def _rotateImageGrid(X, Y, Z, pitch):
    PTS = np.vstack((X.flatten(), Y.flatten(), Z.flatten()))
    PTS = np.matrix(PTS)
    # Pitch transformation
    Tx = linalg.expm( np.matrix([[0,    0,  0],             # rotation via matrix exponential
                                 [0,    0,  pitch],
                                 [0, -pitch, 0]         ]) )
    PTSt = Tx*PTS                                  # rotation w.r.t x-axis about pitch angle
    # reshape back to transformed grid
    oimw = int(max(X.flatten().tolist())*2.0 + 1)
    oimh = int(max(Y.flatten().tolist())*2.0 + 1)
    Xt = PTSt[0].reshape((oimh, oimw))
    Yt = PTSt[1].reshape((oimh, oimw))
    Zt = PTSt[2].reshape((oimh, oimw))
    return Xt, Yt, Zt

def _imageGrid2spherical(X, Y, Z):
    Theta = np.arctan2(X, Z)
    Phi = np.arctan(Y/np.sqrt(np.square(X)+np.square(Z)))
    return Theta, Phi

def _angle2angle(theta):
    '''
    Keeps angle in radians in the interval (0,2*pi)
    '''
    # out of the left bound of pano image
    i,j = np.where(theta < 0)
    for ii, jj in zip(i, j):
        theta[ii, jj] += 2*math.pi

    # out of the right bound of pano image
    i,j = np.where(theta >= 2*math.pi)
    for ii, jj in zip(i, j):
        theta[ii, jj] -= -2*math.pi
    return theta

def _spherical2panoImage(Theta, Phi, iimw, iimh):
    iimcu = iimw/2 - 0.5
    iimcv = iimh/2 - 0.5                 # input image center
    sw = iimw/2/math.pi                # scale pixels per radian
    sh = iimh/math.pi                  # scale pixels per radian
    U = sw*Theta + iimcu
    V = sh*Phi   + iimcv
    U = U%iimw
    return U, V

def _shiftImageGrid(X, Y, Z, pitch):
    '''
    Shift image grid such that image center corresponds to
    the pitched ray
    '''
    f = Z[1,1]                  # focal length in pixels
    offsetY = f*math.tan(pitch)
    Xt = X
    Yt = Y + offsetY
    Zt = Z
    return Xt,Yt,Zt

def main():
    pano_dir = Path(sys.argv[1]).resolve()
    save_dir = Path(sys.argv[2]).resolve()
    assert pano_dir.exists(), f"Panoramas dir does not exist: {pano_dir}"
    save_dir.mkdir(exist_ok=True, parents=True)

    for f in glob.glob(f"{pano_dir}/*.jpg"):
        img = Image.open(f)
        img = np.array(img)
        name = os.path.basename(f).replace(".jpg", "")
        for i, yaw in enumerate([45, 135, 225, 315]):
            img_out = pano2cutout(img=img, yaw=yaw, pitch=-15, cutout_type='rectified')
            save(f'{save_dir}/{name}_{i}.png', img_out)

if __name__=='__main__':
    main()