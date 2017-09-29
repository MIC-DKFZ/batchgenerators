from _warnings import warn

from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop, center_crop_seg, pad, random_crop


def center_crop_generator(generator, output_size):
    warn("using deprecated generator center_crop_generator", Warning)
    '''
    yields center crop of size output_size (may be int or tuple) from data and seg
    '''
    for data_dict in generator:
        assert "data" in list(
            data_dict.keys()), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        seg = None
        if "seg" in list(data_dict.keys()):
            seg = data_dict["seg"]
        data, seg = center_crop(data, output_size, seg)
        data_dict["data"] = data
        if seg is not None:
            data_dict["seg"] = seg
        yield data_dict


def center_crop_seg_generator(generator, output_size):
    warn("using deprecated generator center_crop_seg_generator", Warning)

    '''
    yields center crop of size output_size (from seg (forwards data with size unchanged). This generator is used if the
    output shape of a segmentation network is different from the input shape (f. ex when unpadded convolutions are used)
    '''
    for data_dict in generator:
        do_seg = False
        seg = None
        if "seg" in list(data_dict.keys()):
            seg = data_dict["seg"]
            do_seg = True
        if not do_seg:
            Warning("You used center_crop_seg_generator but there is no 'seg' key in your data_dict")
            yield data_dict
        data_dict["seg"] = center_crop_seg(seg, output_size)
        yield data_dict


def random_crop_generator(generator, crop_size=128, margins=(0, 0, 0)):
    warn("using deprecated generator random_crop_generator", Warning)

    '''
    yields a random crop of size crop_size, crop_size may be a tuple with one entry for each dimension of your data (2D/3D)
    :param margins: allows to give cropping margins measured symmetrically from the image boundaries, which
    restrict the 'box' from which to randomly crop
    '''
    for data_dict in generator:
        assert "data" in list(
            data_dict.keys()), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        seg = None
        if "seg" in list(data_dict.keys()):
            seg = data_dict["seg"]
        data, seg = random_crop(data, seg, crop_size, margins)
        data_dict["data"] = data
        if seg is not None:
            data_dict["seg"] = seg
        yield data_dict


def pad_generator(generator, new_size, pad_value_data=None, pad_value_seg=None):
    warn("using deprecated generator pad_generator", Warning)

    '''
    pads data and seg with value pad_value so that the images have the size new_size
    if pad_value is None then the value of img[0,0] is taken (for each channel in each sample in the minibatch separately), same with seg
    '''
    for data_dict in generator:
        assert "data" in list(
            data_dict.keys()), "your data generator needs to return a python dictionary with at least a 'data' key value pair"
        data = data_dict["data"]
        seg = None
        if "seg" in list(data_dict.keys()):
            seg = data_dict["seg"]
        data, seg = pad(data, new_size, seg, pad_value_data, pad_value_seg)
        if seg is not None:
            data_dict["seg"] = seg
        data_dict["data"] = data
        yield data_dict
