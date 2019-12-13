import numpy as np
from collections import Counter
from lidar_interpolate_toolbox import *
from path import Path
from PIL import Image
from imageio import imread
from visdom import Visdom
from color_utils import *

vis = Visdom()


def read_raw_calib_file(filepath):
    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def generate_depth_map(img, velo, Tr, R_rect, P_rect, depth_size_ratio=1, depth_scale=2.5, img_height=256,
                       img_width=512,
                       choose_closest=False):
    # compute projection matrix velodyne->image plane

    velo2cam = np.vstack((Tr, np.array([0, 0, 0, 1.0])))

    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = R_rect.reshape(3, 3)

    P_rect[0] /= depth_size_ratio
    P_rect[1] /= depth_size_ratio
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, -1:]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1

    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < img_width / depth_size_ratio)
    val_inds = val_inds & (velo_pts_im[:, 1] < img_height / depth_size_ratio)
    velo_pts_im = velo_pts_im[val_inds, :]
    # project to image
    depth = np.zeros((img_height // depth_size_ratio, img_width // depth_size_ratio)).astype(
        np.float32)
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    if choose_closest:
        def sub2ind(matrixSize, rowSub, colSub):
            m, n = matrixSize
            return rowSub * (n - 1) + colSub - 1

        inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
        for dd in dupe_inds:
            pts = np.where(inds == dd)[0]
            x_loc = int(velo_pts_im[pts[0], 0])
            y_loc = int(velo_pts_im[pts[0], 1])
            depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()

    depth[depth < 0] = 0
    depth = np.floor((depth - depth.min()) / (depth.max() - depth.min()) * 255) * depth_scale
    depth[depth > 255.] = 255
    interpolated_lidar = sd_filter(img, depth)
    return interpolated_lidar


def extract_data(root, name, img_height=256, img_width=512):
    root = Path(root)
    img_file = root / 'image_2' / name + '.png'
    img = Image.fromarray(imread(img_file))
    zoom_y = img_height / img.size[1]
    zoom_x = img_width / img.size[0]
    img = img.resize((img_width, img_height))

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    calib_file = root / 'calib' / name + '.txt'
    filedata = read_raw_calib_file(calib_file)
    P_rect = np.reshape(filedata['P2'], (3, 4))
    R_rect = filedata['R0_rect'].reshape(3, 3)

    P_rect[0] *= zoom_x
    P_rect[1] *= zoom_y

    Tr = filedata['Tr_velo_to_cam'].reshape(3, 4)

    velo_name = root / 'velodyne' / name + '.bin'
    velo = np.fromfile(velo_name, dtype=np.float32).reshape(-1, 4)
    velo[:, 3] = 1
    velo = velo[velo[:, 0] >= 0, :]
    depth = generate_depth_map(img, velo, Tr, R_rect, P_rect)
    vis.images(
        array2color(depth, max_value=None,
                    colormap='magma'), win='d', opts=dict(title='d'))
    return array2color(depth, max_value=None, colormap='magma')


if __name__ == '__main__':
    root = Path('./data/object/testing')
    imgs = root / 'image_2'
    files = imgs.files('*.png')
    for f in files:
        depth = extract_data(root, f.stem)
        if 'float' in str(depth.dtype):
            if depth.max() <= 1:
                depth = depth * 255.
            depth = np.uint8(depth)
        depth = np.transpose(depth, (1, 2, 0))
        img = Image.fromarray(depth)
        img.save(root / 'lidar' / f.stem + '.png')

    root = Path('./data/object/training')
    imgs = root / 'image_2'
    files = imgs.files('*.png')
    for f in files:
        depth = extract_data(root, f.stem)
        if 'float' in str(depth.dtype):
            if depth.max() <= 1:
                depth = depth * 255.
            depth = np.uint8(depth)
        depth = np.transpose(depth, (1, 2, 0))
        img = Image.fromarray(depth)
        img.save(root / 'lidar' / f.stem + '.png')
