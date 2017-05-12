import numpy as np

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

def random_crop_generator(generator, crop_size=128, margins=[0,0,0]):
    '''
    yields a random crop of size crop_size, crop_size may be a tuple with one entry for each dimension of your data (2D/3D)
    :param margins: allows to give cropping margins measured symmetrically from the image boundaries, which
    restrict the 'box' from which to randomly crop
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
            lb_x = np.random.randint(margins[0], data.shape[2]-crop_size[0]-margins[0])
        elif crop_size[0] == data.shape[2]:
            lb_x = 0
        else:
            raise ValueError, "crop_size[0] must be smaller or equal to the images x dimension"

        if crop_size[1] < data.shape[3]:
            lb_y = np.random.randint(margins[1], data.shape[3]-crop_size[1]-margins[1])
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
                lb_z = np.random.randint(margins[2], data.shape[4]-crop_size[2]-margins[2])
            elif crop_size[2] == data.shape[4]:
                lb_z = 0
            else:
                raise ValueError, "crop_size[2] must be smaller or equal to the images z dimension"
            data_dict["data"] = data[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1], lb_z:lb_z+crop_size[2]]
            if do_seg:
                data_dict["seg"] = seg[:, :, lb_x:lb_x+crop_size[0], lb_y:lb_y+crop_size[1], lb_z:lb_z+crop_size[2]]
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