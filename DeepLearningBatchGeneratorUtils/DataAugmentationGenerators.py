__author__ = 'Fabian Isensee'

import numpy as np
from scipy.ndimage import interpolation
from scipy.ndimage import map_coordinates
from utils import *
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import resize


def ultimate_transform_generator(generator,
                                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                                 do_rotation=True, angle_x=(0, 2*np.pi), angle_y=(0, 2*np.pi), angle_z = (0, 2*np.pi),
                                 do_scale=True, scale=(0.75, 1.25)):
    '''
    THE ultimate generator. It has all you need:
    :param generator:
    :param do_elastic_deform:
    :param alpha:
    :param sigma:
    :param do_rotation:
    :param angle_x:
    :param angle_y:
    :param angle_z:
    :param do_scale:
    :param scale:
    :return:
    '''
    if not (isinstance(alpha, list) or isinstance(alpha, tuple)):
        alpha = [alpha, alpha]
    if not (isinstance(sigma, list) or isinstance(sigma, tuple)):
        sigma = [sigma, sigma]
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        do_seg = False
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
        shape = np.array(data.shape[2:])
        dim = len(shape)
        for sample_id in xrange(data.shape[0]):
            coords = create_zero_centered_coordinate_mesh(shape)
            if do_elastic_deform:
                a = np.random.uniform(alpha[0], alpha[1])
                s = np.random.uniform(sigma[0], sigma[1])
                coords = elastic_deform_coordinates(coords, a, s)
            if do_rotation:
                a_x = np.random.uniform(angle_x[0], angle_x[1])
                if dim == 3:
                    a_y = np.random.uniform(angle_y[0], angle_y[1])
                    a_z = np.random.uniform(angle_z[0], angle_z[1])
                    coords = rotate_coords_3d(coords, a_x, a_y, a_z)
                else:
                    coords = rotate_coords_2d(coords, a_x)
            if do_scale:
                sc = np.random.uniform(scale[0], scale[1])
                coords = scale_coords(coords, sc)
            coords = uncenter_coords(coords)
            for channel_id in range(data.shape[1]):
                data[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, 3, 'nearest')
            if do_seg:
                for channel_id in range(seg.shape[1]):
                    seg[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, 0, 'constant', cval=0.0)
            yield data_dict


def rotation_and_elastic_transform_generator(generator, alpha=100, sigma=10, angle_x=(0, 2*np.pi), angle_y=(0, 2*np.pi), angle_z = (0, 2*np.pi)):
    '''
    If you plan on using rotations and elastic deformations, use this generator. Most of the computation time is spend
    while interpolating, and that is where this generator offers better performance. Instead of generating deformation
    coordinates, mapping them, then generating rotation coordinates and again do a mapping, this generator will generate
    deform coordinates, rotate the coordinates (which is fast because matrix multiplication) and then interpolate the
    data only once!
    :param generator: incoming generator
    :param alpha: amplitude of deformation
    :param sigma: smoothness of deformation
    :param angle_x: range from which rotation angles are sampled
    :param angle_y: range from which rotation angles are sampled
    :param angle_z: range from which rotation angles are sampled
    :return:
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        do_seg = False
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
        shape = np.array(data.shape[2:])
        coords = tuple([range(i) for i in shape])
        if len(coords) == 2:
            grid = np.array(np.meshgrid(*coords)).transpose(0,2,1)
        elif len(coords) == 3:
            grid = np.array(np.meshgrid(*coords)).transpose(0,2,1,3)
        del coords
        # grid must be centered around (0,0) or (0,0,0)
        for dim in xrange(len(shape)):
            grid[dim] -= (shape.astype(float)/2.).astype(int)[dim]
        if do_seg:
            seg_min = np.min(seg)
            seg_max = np.max(seg)
        for sample_id in xrange(data.shape[0]):
            offsets = []
            for _ in range(len(shape)):
                if isinstance(alpha, list) or isinstance(alpha, tuple):
                    assert len(alpha) == 2, "if alpha is a list/tuple, it must have len=2"
                    a = np.random.random() * (alpha[1] - alpha[0]) + alpha[0]
                else:
                    a = alpha
                if isinstance(sigma, list) or isinstance(sigma, tuple):
                    assert len(sigma) == 2, "if sigma is a list/tuple, it must have len=2"
                    s = np.random.random() * (sigma[1] - sigma[0]) + sigma[0]
                else:
                    s = sigma
                offsets.append(gaussian_filter((np.random.random(shape) * 2 - 1), s, mode="constant", cval=0) * a)
            offsets = np.array(offsets)
            if len(shape) == 3:
                rot = create_random_rotation(angle_x, angle_y, angle_z)
            elif len(shape) == 2:
                rot = create_matrix_rotation_2d(np.random.uniform(*angle_z))
            new_mesh = np.dot(grid.reshape(len(shape), -1).transpose(), rot).transpose().reshape(grid.shape)
            new_mesh = np.array([i+j for i,j in zip(offsets, new_mesh)])
            new_mesh = [i for i in new_mesh]
            del offsets
            # revert centering of grid
            for dim in xrange(len(shape)):
                new_mesh[dim] += (shape.astype(float)/2.).astype(int)[dim]
            for channel_id in xrange(data.shape[1]):
                data[sample_id, channel_id] = map_coordinates(data[sample_id, channel_id], new_mesh, order=3, mode="nearest")
            if do_seg:
                for channel_id in xrange(seg.shape[1]):
                    seg[sample_id, channel_id] = map_coordinates(seg[sample_id, channel_id], new_mesh, order=0, mode="nearest")
            del new_mesh
        if do_seg:
            seg[seg > seg_max] = seg_max
            seg[seg < seg_min] = seg_min
        yield data_dict

def rotation_generator(generator, angle_x = (0, 2*np.pi), angle_y = (0, 2*np.pi), angle_z = (0, 2*np.pi)):
    '''
    Rotates data and seg
    :param generator: incoming generator
    :param angle_x: range from which rotation angles are sampled
    :param angle_y: range from which rotation angles are sampled
    :param angle_z: range from which rotation angles are sampled
    :return:
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        do_seg = False
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
        shape = np.array(data.shape[2:])
        coords = tuple([range(i) for i in shape])
        if len(coords) == 2:
            grid = np.array(np.meshgrid(*coords)).transpose(0,2,1)
        elif len(coords) == 3:
            grid = np.array(np.meshgrid(*coords)).transpose(0,2,1,3)
        # grid must be centered around (0,0) or (0,0,0)
        for dim in xrange(grid.shape[0]):
            grid[dim] -= (shape.astype(float)/2.).astype(int)[dim]
        if do_seg:
            seg_min = np.min(seg)
            seg_max = np.max(seg)
        for sample_id in xrange(data.shape[0]):
            if len(coords) == 3:
                rot = create_random_rotation(angle_x, angle_y, angle_z)
            elif len(coords) == 2:
                rot = create_matrix_rotation_2d(np.random.uniform(*angle_z))
            new_mesh = np.dot(grid.reshape(len(coords), -1).transpose(), rot).transpose().reshape(grid.shape)
            # revert centering of grid
            for dim in xrange(grid.shape[0]):
                new_mesh[dim] += (shape.astype(float)/2.).astype(int)[dim]
            for channel_id in xrange(data.shape[1]):
                data[sample_id, channel_id] = map_coordinates(data[sample_id, channel_id], [i for i in new_mesh], order=3, mode="nearest")
            if do_seg:
                for channel_id in xrange(seg.shape[1]):
                    seg[sample_id, channel_id] = map_coordinates(seg[sample_id, channel_id], [i for i in new_mesh], order=0, mode="nearest")
        if do_seg:
            data_dict["seg"] = np.round(seg)
            data_dict["seg"][seg > seg_max] = seg_max
            data_dict["seg"][seg < seg_min] = seg_min
        yield data_dict

def elastric_transform_generator(generator, alpha=100, sigma=10):
    '''
    yields elastically deformed data and seg
    :param alpha: magnitude of deformation, can be a tuple depicting a range to sample alpha values from
    :param sigma: smoothness of deformation, can be a tuple depicting a range to sample sigma values from
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        do_seg = False
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
            seg_min = np.min(seg)
            seg_max = np.max(seg)
        data_shape = tuple(list(data.shape)[2:])
        for data_idx in xrange(data.shape[0]):
            if isinstance(alpha, list) or isinstance(alpha, tuple):
                assert len(alpha) == 2, "if alpha is a list/tuple, it must have len=2"
                a = np.random.random() * (alpha[1] - alpha[0]) + alpha[0]
            else:
                a = alpha
            if isinstance(sigma, list) or isinstance(sigma, tuple):
                assert len(sigma) == 2, "if sigma is a list/tuple, it must have len=2"
                s = np.random.random() * (sigma[1] - sigma[0]) + sigma[0]
            else:
                s = sigma
            coords = generate_elastic_transform_coordinates(data_shape, a, s)
            for channel_idx in xrange(data.shape[1]):
                data[data_idx, channel_idx] = map_coordinates(data[data_idx, channel_idx], coords, order=3, mode="nearest").reshape(data_shape)
            if do_seg:
                for seg_channel_idx in xrange(seg.shape[1]):
                    seg[data_idx, seg_channel_idx] = map_coordinates(seg[data_idx, seg_channel_idx], coords, order=0, mode="nearest").reshape(seg[data_idx, seg_channel_idx].shape)
        if do_seg:
            data_dict["seg"] = np.round(seg)
            data_dict["seg"][seg > seg_max] = seg_max
            data_dict["seg"][seg < seg_min] = seg_min
        yield data_dict

def center_crop_generator(generator, output_size):
    '''
    yields center crop of size output_size (may be int or tuple) from data and seg
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        do_seg = False
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
        if type(output_size) not in (tuple, list):
            center_crop = [int(output_size)]*(len(data.shape)-2)
        else:
            center_crop = output_size
            assert len(center_crop) == len(data.shape)-2, "If you provide a list/tuple as center crop make sure it has the same dimension as your data (2d/3d)"
        center = np.array(data.shape[2:])/2
        if len(data.shape) == 5:
            data_dict["data"] = data[:, :, int(center[0]-center_crop[0]/2.):int(center[0]+center_crop[0]/2.), int(center[1]-center_crop[1]/2.):int(center[1]+center_crop[1]/2.), int(center[2]-center_crop[2]/2.):int(center[2]+center_crop[2]/2.)]
            if do_seg:
                data_dict["seg"] = seg[:, :, int(center[0]-center_crop[0]/2.):int(center[0]+center_crop[0]/2.), int(center[1]-center_crop[1]/2.):int(center[1]+center_crop[1]/2.), int(center[2]-center_crop[2]/2.):int(center[2]+center_crop[2]/2.)]
        elif len(data.shape) == 4:
            data_dict["data"] = data[:, :, int(center[0]-center_crop[0]/2.):int(center[0]+center_crop[0]/2.), int(center[1]-center_crop[1]/2.):int(center[1]+center_crop[1]/2.)]
            if do_seg:
                data_dict["seg"] = seg[:, :, int(center[0]-center_crop[0]/2.):int(center[0]+center_crop[0]/2.), int(center[1]-center_crop[1]/2.):int(center[1]+center_crop[1]/2.)]
        else:
            raise Exception("Invalid dimension for seg. seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
        yield data_dict

def center_crop_seg_generator(generator, output_size):
    '''
    yields center crop of size output_size (from seg (forwards data with size unchanged). This generator is used if the
    output shape of a segmentation network is different from the input shape (f. ex when unpadded convolutions are used)
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        do_seg = False
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
        if not do_seg:
            Warning("You used center_crop_seg_generator but there is no 'seg' key in your data_dict")
            yield data_dict
        if type(output_size) not in (tuple, list):
            center_crop = [int(output_size)]*(len(data.shape)-2)
        else:
            center_crop = output_size
            assert len(center_crop) == len(data.shape)-2, "If you provide a list/tuple as center crop make sure it has the same dimension as your data (2d/3d)"
        center = np.array(data.shape[2:])/2
        if len(seg.shape) == 4:
            data_dict["seg"] = seg[:, :, int(center[0]-center_crop[0]/2.):int(center[0]+center_crop[0]/2.), int(center[1]-center_crop[1]/2.):int(center[1]+center_crop[1]/2.)]
        elif len(seg.shape) == 5:
            data_dict["seg"] = seg[:, :, int(center[0]-center_crop[0]/2.):int(center[0]+center_crop[0]/2.), int(center[1]-center_crop[1]/2.):int(center[1]+center_crop[1]/2.), int(center[2]-center_crop[2]/2.):int(center[2]+center_crop[2]/2.)]
        else:
            raise Exception("Invalid dimension for seg. seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
        yield data_dict

def mirror_axis_generator(generator):
    '''
    yields mirrored data and seg.
    3D data: 12.5% of each constellation: x only, y only, z only, xy, xz, yz, xzy, none
    2D data: 25% of each constellation: x, xy, y, none
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        do_seg = False
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
        if (len(data.shape) != 4) and (len(data.shape) != 5):
            raise Exception("Invalid dimension for data and seg. data and seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
        BATCH_SIZE = data.shape[0]
        idx = np.arange(BATCH_SIZE)
        for id in idx:
            if np.random.uniform() < 0.5:
                data[id, :, :] = data[id, :, ::-1]
                if do_seg:
                    seg[id, :, :] = seg[id, :, ::-1]
            if np.random.uniform() < 0.5:
                data[id, :, :, :] = data[id, :, :, ::-1]
                if do_seg:
                    seg[id, :, :, :] = seg[id, :, :, ::-1]
            if len(data.shape) == 5:
                if np.random.uniform() < 0.5:
                    data[id, :, :, :, :] = data[id, :, :, :, ::-1]
                    if do_seg:
                        seg[id, :, :, :, :] = seg[id, :, :, :, ::-1]
        yield data_dict

def rescale_and_crop_generator(generator, scale_range, crop_size, random=True):
    '''
    generates crops of zoom ins and outs of the original data. zoom factor can be given in scale_range.
    scale_range should be tuple of len 2. new_shape is determined as old_shape * scale_factor (<1 for zooming out and >1 for zooming in)
    crop size must be len(dim(img))
    random=True -> random crops, False-> center crops

    @ToDo: We could allow different scale factors for each dimension

    ATTENTION: We do not check whether rescaing creates images too small for cropping with crop_size. It is your
    responsibility to see that after zooming out your crop size can be cropped (only if you zoom out ofc)
    '''
    assert len(scale_range) == 2, "scale_range must be a tuple of len 2"
    if scale_range[0] < scale_range[1]:
        scale_range = (scale_range[1], scale_range[0])

    if random:
        crop3d = random_crop_3D_image_batched
        crop2d = random_crop_2D_image_batched
    else:
        crop3d = center_crop_3D_image_batched
        crop2d = center_crop_2D_image_batched

    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        shp = data.shape[2:]
        do_seg = False
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True

        if len(shp) == 2:
            new_data = np.zeros((data.shape[0], data.shape[1], crop_size[0], crop_size[1]), dtype=data.dtype)
            if do_seg:
                new_seg = np.zeros((seg.shape[0], seg.shape[1], crop_size[0], crop_size[1]), dtype=seg.dtype)
        if len(shp) == 3:
            new_data = np.zeros((data.shape[0], data.shape[1], crop_size[0], crop_size[1], crop_size[2]), dtype=data.dtype)
            if do_seg:
                new_seg = np.zeros((seg.shape[0], seg.shape[1], crop_size[0], crop_size[1], crop_size[2]), dtype=seg.dtype)

        for b in range(data.shape[0]):
            # draw a random scale factor
            scale = np.random.uniform(scale_range[0], scale_range[1])
            new_shp = np.round(np.array(shp) * scale).astype(int)

            # create some temporary variables to store the rescaled version of each each training sample
            if len(new_shp) == 2:
                new_data_tmp = np.zeros((data.shape[1], new_shp[0], new_shp[1]), dtype=data.dtype)
                if do_seg:
                    new_seg_tmp = np.zeros((seg.shape[1], new_shp[0], new_shp[1]), dtype=seg.dtype)
            if len(new_shp) == 3:
                new_data_tmp = np.zeros((data.shape[1], new_shp[0], new_shp[1], new_shp[2]), dtype=data.dtype)
                if do_seg:
                    new_seg_tmp = np.zeros((seg.shape[1], new_shp[0], new_shp[1], new_shp[2]), dtype=seg.dtype)

            # rescale the images (do each color channel separately)
            for c in range(data.shape[1]):
                new_data_tmp[c] = resize(data[b, c], new_shp, 3, 'edge', preserve_range=True)
            if do_seg:
                for c in range(seg.shape[1]):
                    new_seg_tmp[c] = resize(seg[b, c], new_shp, 0, 'edge', preserve_range=True)

            # now crop
            if len(new_shp) == 2:
                if do_seg:
                    tmp = crop2d(np.vstack((new_data_tmp, new_seg_tmp))[None], crop_size)[0]
                    new_data[b] = tmp[:data.shape[1]]
                    new_seg[b] = tmp[data.shape[1]:]
                else:
                    new_data[b] = crop2d(new_data_tmp[None], crop_size)[0]
            if len(new_shp) == 3:
                if do_seg:
                    tmp = crop3d(np.vstack((new_data_tmp, new_seg_tmp))[None], crop_size)[0]
                    new_data[b] = tmp[:data.shape[1]]
                    new_seg[b] = tmp[data.shape[1]:]
                else:
                    new_data[b] = crop3d(new_data_tmp[None], crop_size)[0]

        data_dict['data'] = new_data
        if do_seg:
            data_dict['seg'] = new_seg
        yield data_dict

def random_crop_generator(generator, crop_size=128):
    '''
    yields a random crop of size crop_size, crop_size may be a tuple with one entry for each dimension of your data (2D/3D)
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        do_seg = False
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
        if type(crop_size) not in (tuple, list):
            crop_size = [crop_size]*(len(data.shape)-2)
        else:
            assert len(crop_size) == len(data.shape)-2, "If you provide a list/tuple as center crop make sure it has the same dimension as your data (2d/3d)"

        if crop_size[0] < data.shape[2]:
            lb_x = np.random.randint(0, data.shape[2]-crop_size[0])
        elif crop_size[0] == data.shape[2]:
            lb_x = 0
        else:
            raise ValueError, "crop_size[0] must be smaller or equal to the images x dimension"

        if crop_size[1] < data.shape[3]:
            lb_y = np.random.randint(0, data.shape[3]-crop_size[1])
        elif crop_size[1] == data.shape[3]:
            lb_y = 0
        else:
            raise ValueError, "crop_size[1] must be smaller or equal to the images y dimension"

        if len(data.shape) == 4:
            data_dict["data"] = data[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1]]
            if do_seg:
                data_dict["seg"] = seg[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1]]
        elif len(data.shape) == 5:
            if crop_size[2] < data.shape[4]:
                lb_z = np.random.randint(0, data.shape[4]-crop_size[2])
            elif crop_size[2] == data.shape[4]:
                lb_z = 0
            else:
                raise ValueError, "crop_size[2] must be smaller or equal to the images z dimension"
            data_dict["data"] = data[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1], lb_z:lb_z+crop_size[2]]
            if do_seg:
                data_dict["seg"] = seg[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1], lb_z:lb_z+crop_size[2]]
        yield data_dict

def data_channel_selection_generator(generator, selected_channels):
    '''
    yields selected channels from data
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        data_dict["data"] = data[:, selected_channels]
        yield data_dict

def seg_channel_selection_generator(generator, selected_channels, keep_discarded_seg=False):
    '''
    yields selected channels from seg
    '''
    for data_dict in generator:
        do_seg = False
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
        if not do_seg:
            Warning("You used center_crop_seg_generator but there is no 'seg' key in your data_dict, returning data_dict unmodified")
            yield data_dict
        data_dict["seg"] = seg[:, selected_channels]
        if keep_discarded_seg:
            discarded_seg_idx = [i for i in range(len(seg[0])) if i not in selected_channels]
            data_dict['discarded_seg'] = seg[:, discarded_seg_idx]
        yield data_dict

def pad_generator(generator, new_size, pad_value_data=None, pad_value_seg=None):
    '''
    pads data and seg with value pad_value so that the images have the size new_size
    if pad_value is None then the value of img[0,0] is taken (for each channel in each sample in the minibatch separately), same with seg
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        do_seg = False
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
            res_seg = np.ones([seg.shape[0], seg.shape[1]]+list(new_size), dtype=seg.dtype)*pad_value_seg
        shape = tuple(list(data.shape)[2:])
        start = np.array(new_size)/2. - np.array(shape)/2.
        res_data = np.ones([data.shape[0], data.shape[1]]+list(new_size), dtype=data.dtype)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if pad_value_data is None:
                    if len(shape) == 2:
                        pad_value_tmp = data[i, j, 0, 0]
                    elif len(shape) == 3:
                        pad_value_tmp = data[i, j, 0, 0, 0]
                    else:
                        raise Exception("Invalid dimension for data and seg. data and seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
                else:
                    pad_value_tmp = pad_value_data
                res_data[i, j] = pad_value_tmp
                if len(shape) == 2:
                    res_data[i, j, int(start[0]):int(start[0])+int(shape[0]), int(start[1]):int(start[1])+int(shape[1])] = data[i, j]
                elif len(shape) == 3:
                    res_data[i, j, int(start[0]):int(start[0])+int(shape[0]), int(start[1]):int(start[1])+int(shape[1]), int(start[2]):int(start[2])+int(shape[2])] = data[i, j]
                else:
                    raise Exception("Invalid dimension for data and seg. data and seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
                if do_seg:
                    for j in range(seg.shape[1]):
                        if pad_value_seg is None:
                            if len(shape) == 2:
                                pad_value_tmp = seg[i, j, 0, 0]
                            elif len(shape) == 3:
                                pad_value_tmp = seg[i, j, 0, 0, 0]
                            else:
                                raise Exception(
                                    "Invalid dimension for data and seg. data and seg should be either [BATCH_SIZE, channels, x, y] or [BATCH_SIZE, channels, x, y, z]")
                        else:
                            pad_value_tmp = pad_value_seg
                        res_seg[i, j] = pad_value_tmp
                        if len(shape) == 2:
                            res_seg[i, j, int(start[0]):int(start[0])+int(shape[0]), int(start[1]):int(start[1])+int(shape[1])] = seg[i, j]
                        elif len(shape) == 3:
                            res_seg[i, j, int(start[0]):int(start[0])+int(shape[0]), int(start[1]):int(start[1])+int(shape[1]), int(start[2]):int(start[2])+int(shape[2])] = seg[i, j]
        if do_seg:
            data_dict["seg"] = res_seg
        data_dict["data"] = res_data
        yield data_dict

