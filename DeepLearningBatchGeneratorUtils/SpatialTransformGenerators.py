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
                                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                                 border_mode_seg='constant', border_cval_seg=0, order_seg=0):
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
                data[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data, border_mode_data, cval=border_cval_data)
            if do_seg:
                for channel_id in range(seg.shape[1]):
                    seg[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg, border_mode_seg, cval=border_cval_seg)
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


def mirror_axis_generator(generator, axes=[2,3,4]):
    '''
    yields mirrored data and seg.
    iff axes == [2,3,4]:
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
            if 2 in axes and np.random.uniform() < 0.5:
                data[id, :, :] = data[id, :, ::-1]
                if do_seg:
                    seg[id, :, :] = seg[id, :, ::-1]
            if 3 in axes and np.random.uniform() < 0.5:
                data[id, :, :, :] = data[id, :, :, ::-1]
                if do_seg:
                    seg[id, :, :, :] = seg[id, :, :, ::-1]
            if 4 in axes and len(data.shape) == 5:
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
    Warning("This Generator is deprecated and will be removed soon. Please use ultimate_transform_generator instead")
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


def channel_translation_generator(generator, const_channel=0, max_shifts={'z':2, 'y':2, 'x':2}):
    """
    Translates all channels within an instance of a batch according to randomly drawn shifts from within [-max_shift, max_shift].
    One channel is held constant, the others are shifted in the same manner.
    :param generator:
    :param const_channel:
    :param max_shifts:
    :return:
    """

    for data_dict in generator:

        data = data_dict["data"]
        shape = data.shape

        const_data = data[:,[const_channel]]
        trans_data = data[:,[i for i in range(shape[1]) if i != const_channel]]

        # iterate the batch dimension
        for j in range(shape[0]):

            slice = trans_data[j]

            ixs = {}
            pad = {}

            if len(shape) == 5:
                dims = ['z', 'y', 'x']
            else:
                dims = ['y', 'x']

            # iterate the image dimensions, randomly draw shifts/translations
            for i,v in enumerate(dims):
                rand_shift = np.random.choice(range(-max_shifts[v], max_shifts[v], 1))

                if rand_shift > 0:
                    ixs[v] = {'lo':0, 'hi':-rand_shift}
                    pad[v] = {'lo':rand_shift, 'hi':0}
                else:
                    ixs[v] = {'lo':abs(rand_shift), 'hi':shape[2+i]}
                    pad[v] = {'lo':0, 'hi':abs(rand_shift)}

            # shift and pad so as to retain the original image shape
            if len(shape) == 5:
                slice = slice[:,ixs['z']['lo']:ixs['z']['hi'],ixs['y']['lo']:ixs['y']['hi'],ixs['x']['lo']:ixs['x']['hi']]
                slice = np.pad(slice, ((0,0),(pad['z']['lo'], pad['z']['hi']), (pad['y']['lo'], pad['y']['hi']), (pad['x']['lo'], pad['x']['hi'])), \
					 mode='constant', constant_values=(0, 0))
            if len(shape) == 4:
                slice = slice[:, ixs['y']['lo']:ixs['y']['hi'],ixs['x']['lo']:ixs['x']['hi']]
                slice = np.pad(slice, ((0,0), (pad['y']['lo'], pad['y']['hi']), (pad['x']['lo'], pad['x']['hi'])), \
					 mode='constant', constant_values=(0, 0))

            trans_data[j] = slice

        data_dict['data'] = np.concatenate([const_data, trans_data], axis=1)

        yield data_dict


def ultimate_transform_generator_v2(generator, patch_size, patch_center_dist_from_border=30,
                                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                                 do_rotation=True, angle_x=(0, 2*np.pi), angle_y=(0, 2*np.pi), angle_z = (0, 2*np.pi),
                                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                                 border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True):
    '''
    THE ultimate generator. It has all you need. It alleviates the problem of having to crop your data to a reasonably sized patch size before plugging it into the
    old ultimate_transform generator (In the old one you would put in patches larger than your final patch size so that rotations and deformations to not introduce black borders).
    Before: Large crops = no borders but slow, small crops = black borders (duh).
    Here you can just plug in the whole uncropped image and get your desired patch size as output, without performance loss or black borders
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
        shape = patch_size
        assert len(shape) == len(data.shape[2:]), "dimension of patch_size and data must match!"
        dim = len(shape)
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
            if dim == 2:
                seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
            else:
                seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]), dtype=np.float32)
        if dim == 2:
            data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1]), dtype=np.float32)
        else:
            data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]), dtype=np.float32)

        if not isinstance(patch_center_dist_from_border, (list, tuple)):
            patch_center_dist_from_border = dim * [patch_center_dist_from_border]
        for sample_id in xrange(data.shape[0]):
            coords = create_zero_centered_coordinate_mesh(shape)
            if do_elastic_deform:
                a = np.random.uniform(alpha[0], alpha[1])
                s = np.random.uniform(sigma[0], sigma[1])
                coords = elastic_deform_coordinates(coords, a, s)
            if do_rotation:
                if angle_x[0] == angle_x[1]:
                    a_x = angle_x[0]
                else:
                    a_x = np.random.uniform(angle_x[0], angle_x[1])
                if dim == 3:
                    if angle_y[0] == angle_y[1]:
                        a_y = angle_y[0]
                    else:
                        a_y = np.random.uniform(angle_y[0], angle_y[1])
                    if angle_z[0] == angle_z[1]:
                        a_z = angle_z[0]
                    else:
                        a_z = np.random.uniform(angle_z[0], angle_z[1])
                    coords = rotate_coords_3d(coords, a_x, a_y, a_z)
                else:
                    coords = rotate_coords_2d(coords, a_x)
            if do_scale:
                if np.random.random() < 0.5 and scale[0] < 1:
                    sc = np.random.uniform(scale[0], 1)
                else:
                    sc = np.random.uniform(max(scale[0], 1), scale[1])
                coords = scale_coords(coords, sc)
            # now find a nice center location
            for d in range(dim):
                if random_crop:
                    ctr = np.random.uniform(patch_center_dist_from_border[d], data.shape[d+2]-patch_center_dist_from_border[d])
                else:
                    ctr = int(np.round(data.shape[d+2] / 2.))
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order_data, border_mode_data, cval=border_cval_data)
            if do_seg:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order_seg, border_mode_seg, cval=border_cval_seg)
        if do_seg:
            data_dict['seg'] = seg_result
        data_dict['data'] = data_result
        yield data_dict