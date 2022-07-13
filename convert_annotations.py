import os
import sys
import numpy
import skimage.io
import skimage.transform
import open3d as o3d
import pyflann
import xml.etree.ElementTree as ET
from util.class_util import class_names, class_colors

import argparse

parser = argparse.ArgumentParser("")
parser.add_argument( '--base-dir', type=str, required=True, help='')
parser.add_argument( '--pcd-dir', type=str, default=None, help='')
parser.add_argument( '--images-dir', type=str, default=None, help='')
parser.add_argument( '--annotations-dir', type=str, default=None, help='')
parser.add_argument( '--camera-file', type=str, default=None, help='')
parser.add_argument( '--output-height', type=int, default=256, help='')
parser.add_argument( '--output-width', type=int, default=256, help='')
FLAGS = parser.parse_args()

label_id_map = {j:i for i,j in enumerate(class_names)}
print('label_id_map', label_id_map)
base_dir = FLAGS.base_dir

label_id_map = {j:i for i,j in enumerate(class_names)}
print('label_id_map', label_id_map)
image_dir = base_dir + "images"

# Load point cloud data
pcd_data = []
pcd_labels = []
load_from_npy = True
pcd_annotations_folder = base_dir + '/3D_annotation'
for i in sorted(os.listdir(pcd_annotations_folder)):
    pcd = o3d.io.read_point_cloud(pcd_annotations_folder + '/' + i)
    xyz = numpy.asarray(pcd.points)
    classname = i.split('.')[0]
    pcd_labels.append(label_id_map[classname])
    pcd_data.append(xyz)
    print('Loaded', i, xyz.shape)
#    break

 # Assume exported from Pix4D
f = open(base_dir + '/params/location2-60degrees-only_calibrated_camera_parameters.txt', 'r')
for i in range(8):
    f.readline()
intrinsics = []
extrinsics = []
image_names = []
while True:
    l = f.readline()
    if not l:
        break
    image_names.append(l.split()[0])
    w, h = [int(t) for t in l.split()[1:]]
    focal, _, cx = [float(t) for t in f.readline().split()]
    cy = float(f.readline().split()[2])
    intrinsics.append([focal, 0, 0])
    for i in range(3):
        f.readline()
    R = numpy.zeros((3,3))
    t = numpy.array([float(j) for j in f.readline().split()])
    R[0, :] = [float(j) for j in f.readline().split()]
    R[1, :] = [float(j) for j in f.readline().split()]
    R[2, :] = [float(j) for j in f.readline().split()]
    extrinsics.append(numpy.eye(4))
    extrinsics[-1][:3, :3] = R
    extrinsics[-1][:3, 3] = -R.dot(t)
intrinsics = numpy.array(intrinsics)[numpy.argsort(image_names)]
extrinsics = numpy.array(extrinsics)[numpy.argsort(image_names)]
f.close()
cx_offset = cx - (w-1)*0.5
cy_offset = cy - (h-1)*0.5
flip_u = True
print('Loaded camera coords', FLAGS.camera_file)

# Utility function to fill in empty pixels using nearest neighbor interpolation
def fill_empty(I):
    interpolated = I.copy()
    mask = I.mean(axis=2) > 0
    if mask.sum()==0:
        return interpolated
    v,u = numpy.nonzero(mask)
    x = numpy.transpose((v,u))
    flann = pyflann.FLANN()
    pstack = numpy.transpose(numpy.nonzero(numpy.logical_not(mask)))
    q,_ = flann.nn(x.astype(numpy.int32), pstack.astype(numpy.int32), 1, algorithm='kdtree_simple')
    for i in range(len(pstack)):
        min_v,min_u = x[q[i]]
        M = I[min_v, min_u, :]
        interpolated[pstack[i,0], pstack[i,1], :] = M
    return interpolated

# Write image annotations
images_folder = base_dir + '/images'
image_annotations_folder = base_dir + '/image_annotations'
if not os.path.exists(image_annotations_folder):
    os.mkdir(image_annotations_folder)
image_height = None
image_width = None
image_downsample = 8
camera_id = 0
for i in sorted(os.listdir(images_folder)):
    if not i.endswith('.JPG') and not i.endswith('.jpg'):
        continue
    if image_names is not None and not i in image_names:
        continue
    if image_height is None:
        I = skimage.io.imread(images_folder + '/' + i)
        image_height = I.shape[0]
        image_width = I.shape[1]
        cx = (image_width-1)*0.5 + cx_offset
        cy = (image_height-1)*0.5 + cy_offset
        print('Image dimensions', image_width, image_height, cx, cy)

    labels = numpy.zeros((image_height // image_downsample, image_width // image_downsample), dtype=int)
    labels[:] = -1
    R = extrinsics[camera_id, :3, :3]
    t = extrinsics[camera_id, :3, 3]
    f, k1, k2 = intrinsics[camera_id, :]
    k1 = k2 = 0
    for pcd_id in range(len(pcd_data)):
        xyz = pcd_data[pcd_id]
        xyz = xyz.dot(R.T) + t
        xy = xyz[:, :2] / xyz[:, 2:3]
        rp = 1.0 + k1 * (xy**2).sum(axis=1) + k2 * (xy**2).sum(axis=1)**2
        xy = rp.reshape(-1, 1) * xy
#        print(xy.shape, xy.min(axis=0), xy.max(axis=0))
        xy = f * xy
        if flip_u:
            xy[:, 0] = -xy[:, 0]
        xy += [cx, cy]
        xy /= image_downsample
        xy = numpy.round(xy).astype(numpy.int32)

        valid = xy[:, 0] >= 0
        valid = numpy.logical_and(valid, xy[:, 0] < image_width // image_downsample)
        valid = numpy.logical_and(valid, xy[:, 1] >= 0)
        valid = numpy.logical_and(valid, xy[:, 1] < image_height // image_downsample)
#        print('valid', numpy.sum(valid))
        xy = xy[valid, :]
        labels[xy[:, 1], xy[:, 0]] = pcd_labels[pcd_id]

    image_data = class_colors[labels, :]
    image_data[labels==-1] = 0
    image_data = fill_empty(image_data)

    original_image = skimage.io.imread(images_folder + '/' + i)
    original_image = skimage.transform.resize(original_image, (image_height // image_downsample, image_width // image_downsample))
    overlaid_image = (original_image * 0.5 + image_data / 255.0 * 0.5)
    skimage.io.imsave(image_annotations_folder + '/overlaid_' + i, (overlaid_image*255).astype(numpy.uint8))
    
    image_data = skimage.transform.resize(image_data, (FLAGS.output_height, FLAGS.output_width), order=0, preserve_range=True, anti_aliasing=False)
    output_image = image_annotations_folder + '/' + i
    output_image = output_image.replace('.JPG', '.png').replace('.jpg','.png')
    skimage.io.imsave(output_image, image_data.astype(numpy.uint8))
    print('Saved',len(set(labels.flatten())), 'labels to', output_image)

    camera_id += 1
#    if camera_id == 1:
#        break
