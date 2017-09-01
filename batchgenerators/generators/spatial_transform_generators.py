from batchgenerators.augmentations.spatial_transformations import augment_mirroring, augment_channel_translation, augment_spatial


def mirror_axis_generator(generator, axes=(2, 3, 4)):
    '''
    yields mirrored data and seg.
    iff axes == [2,3,4]:
    3D data: 12.5% of each constellation: x only, y only, z only, xy, xz, yz, xzy, none
    2D data: 25% of each constellation: x, xy, y, none
    '''
    for data_dict in generator:
        assert "data" in data_dict.keys(), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        seg = None
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
        data, seg = augment_mirroring(data, seg, axes)
        data_dict["data"] = data
        if seg is not None:
            data_dict["seg"] = seg
        yield data_dict


def channel_translation_generator(generator, const_channel=0, max_shifts=None):
    """
    Translates all channels within an instance of a batch according to randomly drawn shifts from within [-max_shift, max_shift].
    One channel is held constant, the others are shifted in the same manner.
    :param generator:
    :param const_channel:
    :param max_shifts:
    :return:
    """

    if max_shifts is None:
        max_shifts = {'z':2, 'y':2, 'x':2}

    for data_dict in generator:
        data_dict["data"] = augment_channel_translation(data_dict["data"], const_channel, max_shifts)
        yield data_dict


def spatial_augmentation_generator(generator, patch_size, patch_center_dist_from_border=30,
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
        if "seg" in data_dict.keys():
            seg = data_dict["seg"]
            do_seg = True
        data_result, seg_result = augment_spatial(data, seg, patch_size, patch_center_dist_from_border,
                                                  do_elastic_deform, alpha, sigma, do_rotation, angle_x, angle_y,
                                                  angle_z, do_scale, scale, border_mode_data, border_cval_data,
                                                  order_data, border_mode_seg, border_cval_seg, order_seg,
                                                  random_crop)
        if do_seg:
            data_dict['seg'] = seg_result
        data_dict['data'] = data_result
        yield data_dict