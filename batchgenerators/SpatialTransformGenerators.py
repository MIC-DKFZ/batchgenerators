from utils import *


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