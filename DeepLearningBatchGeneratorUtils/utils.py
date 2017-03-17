__author__ = 'fabian'
from scipy.ndimage.filters import gaussian_filter
import numpy as np

def generate_elastic_transform_coordinates(shape, alpha, sigma):
    n_dim = len(shape)
    offsets = []
    for _ in range(n_dim):
        offsets.append(gaussian_filter((np.random.random(shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha)
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.meshgrid(*tmp, indexing='ij')
    indices = [np.reshape(i+j, (-1, 1)) for i,j in zip(offsets, coords)]
    return indices

def generate_noise(shape, alpha, sigma):
    noise = np.random.random(shape) * 2 - 1
    noise = gaussian_filter(noise, sigma, mode="constant", cval=0) *alpha
    return noise

def find_entries_in_array(entries, myarray):
    entries = np.array(entries)
    values = np.arange(np.max(myarray) + 1)
    lut = np.zeros(len(values),'bool')
    lut[entries.astype("int")] = True
    return np.take(lut, myarray.astype(int))

def center_crop_3D_image(img, crop_size):
    center = np.array(img.shape) / 2
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * len(img.shape)
    else:
        center_crop = crop_size
        assert len(center_crop) == len(img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"
    return img[int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.), int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.), int(center[2] - center_crop[2] / 2.):int(center[2] + center_crop[2] / 2.)]

def center_crop_3D_image_batched(img, crop_size):
    # dim 0 is batch, dim 1 is channel, dim 2, 3 and 4 are x y z
    center = np.array(img.shape[2:]) / 2
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * (len(img.shape) -2)
    else:
        center_crop = crop_size
        assert len(center_crop) == (len(img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"
    return img[:, :int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.), int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.), int(center[2] - center_crop[2] / 2.):int(center[2] + center_crop[2] / 2.)]

def center_crop_2D_image(img, crop_size):
    center = np.array(img.shape) / 2
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * len(img.shape)
    else:
        center_crop = crop_size
        assert len(center_crop) == len(img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"
    return img[int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.), int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]

def center_crop_2D_image_batched(img, crop_size):
    # dim 0 is batch, dim 1 is channel, dim 2 and 3 are x y
    center = np.array(img.shape[2:]) / 2
    if type(crop_size) not in (tuple, list):
        center_crop = [int(crop_size)] * (len(img.shape) -2)
    else:
        center_crop = crop_size
        assert len(center_crop) == (len(img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"
    return img[:, :int(center[0] - center_crop[0] / 2.):int(center[0] + center_crop[0] / 2.), int(center[1] - center_crop[1] / 2.):int(center[1] + center_crop[1] / 2.)]

def random_crop_3D_image(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * len(img.shape)
    else:
        assert len(crop_size) == len(img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    if crop_size[0] < img.shape[0]:
        lb_x = np.random.randint(0, img.shape[0] - crop_size[0])
    elif crop_size[0] == img.shape[0]:
        lb_x = 0
    else:
        raise ValueError, "crop_size[0] must be smaller or equal to the images x dimension"

    if crop_size[1] < img.shape[1]:
        lb_y = np.random.randint(0, img.shape[1] - crop_size[1])
    elif crop_size[1] == img.shape[1]:
        lb_y = 0
    else:
        raise ValueError, "crop_size[1] must be smaller or equal to the images y dimension"

    if crop_size[2] < img.shape[2]:
        lb_z = np.random.randint(0, img.shape[2] - crop_size[2])
    elif crop_size[2] == img.shape[2]:
        lb_z = 0
    else:
        raise ValueError, "crop_size[2] must be smaller or equal to the images z dimension"

    return img[lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]

def random_crop_3D_image_batched(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 2)
    else:
        assert len(crop_size) == (len(img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (3d)"

    if crop_size[0] < img.shape[2]:
        lb_x = np.random.randint(0, img.shape[2] - crop_size[0])
    elif crop_size[0] == img.shape[2]:
        lb_x = 0
    else:
        raise ValueError, "crop_size[0] must be smaller or equal to the images x dimension"

    if crop_size[1] < img.shape[3]:
        lb_y = np.random.randint(0, img.shape[3] - crop_size[1])
    elif crop_size[1] == img.shape[3]:
        lb_y = 0
    else:
        raise ValueError, "crop_size[1] must be smaller or equal to the images y dimension"

    if crop_size[2] < img.shape[4]:
        lb_z = np.random.randint(0, img.shape[4] - crop_size[2])
    elif crop_size[2] == img.shape[4]:
        lb_z = 0
    else:
        raise ValueError, "crop_size[2] must be smaller or equal to the images z dimension"

    return img[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]]

def random_crop_2D_image(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * len(img.shape)
    else:
        assert len(crop_size) == len(
            img.shape), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"

    if crop_size[0] < img.shape[0]:
        lb_x = np.random.randint(0, img.shape[0] - crop_size[0])
    elif crop_size[0] == img.shape[0]:
        lb_x = 0
    else:
        raise ValueError, "crop_size[0] must be smaller or equal to the images x dimension"

    if crop_size[1] < img.shape[1]:
        lb_y = np.random.randint(0, img.shape[1] - crop_size[1])
    elif crop_size[1] == img.shape[1]:
        lb_y = 0
    else:
        raise ValueError, "crop_size[1] must be smaller or equal to the images y dimension"

    return img[lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]

def random_crop_2D_image_batched(img, crop_size):
    if type(crop_size) not in (tuple, list):
        crop_size = [crop_size] * (len(img.shape) - 2)
    else:
        assert len(crop_size) == (len(img.shape) - 2), "If you provide a list/tuple as center crop make sure it has the same len as your data has dims (2d)"

    if crop_size[0] < img.shape[2]:
        lb_x = np.random.randint(0, img.shape[2] - crop_size[0])
    elif crop_size[0] == img.shape[2]:
        lb_x = 0
    else:
        raise ValueError, "crop_size[0] must be smaller or equal to the images x dimension"

    if crop_size[1] < img.shape[3]:
        lb_y = np.random.randint(0, img.shape[3] - crop_size[1])
    elif crop_size[1] == img.shape[3]:
        lb_y = 0
    else:
        raise ValueError, "crop_size[1] must be smaller or equal to the images y dimension"

    return img[:, :, lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]

def resize_image_by_padding(image, new_shape, pad_value=None):
    shape = tuple(list(image.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2,len(shape))), axis=0))
    if pad_value is None:
        if len(shape)==2:
            pad_value = image[0,0]
        elif len(shape)==3:
            pad_value = image[0, 0, 0]
        else:
            raise ValueError("Image must be either 2 or 3 dimensional")
    '''pad_x = [(new_shape[0]-shape[0])/2, (new_shape[0]-shape[0])/2]
    pad_y = [(new_shape[1]-shape[1])/2, (new_shape[1]-shape[1])/2]
    if len(shape) == 3:
        pad_z = [(new_shape[2]-shape[2])/2, (new_shape[2]-shape[2])/2]
    if (new_shape[0]-shape[0])%2 == 1:
        pad_x[1] += 1
    if (new_shape[1]-shape[1])%2 == 1:
        pad_y[1] += 1
    if (len(shape) == 3) and ((new_shape[1]-shape[1])%2 == 1):
        pad_z[1] += 1'''
    res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
    start = np.array(new_shape)/2. - np.array(shape)/2.
    if len(shape) == 2:
        res[int(start[0]):int(start[0])+int(shape[0]), int(start[1]):int(start[1])+int(shape[1])] = image
    elif len(shape) == 3:
        res[int(start[0]):int(start[0])+int(shape[0]), int(start[1]):int(start[1])+int(shape[1]), int(start[2]):int(start[2])+int(shape[2])] = image
    return res

def create_matrix_rotation_x_3d(angle, matrix = None):
    rotation_x = np.array([[1,              0,              0],
                           [0,              np.cos(angle),  -np.sin(angle)],
                           [0,              np.sin(angle),  np.cos(angle)]])
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix = None):
    rotation_y = np.array([[np.cos(angle),  0,              np.sin(angle)],
                           [0,              1,              0],
                           [-np.sin(angle), 0,              np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix = None):
    rotation_z = np.array([[np.cos(angle),  -np.sin(angle), 0],
                           [np.sin(angle),  np.cos(angle),  0],
                           [0,              0,              1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)

def create_matrix_rotation_2d(angle, matrix = None):
    rotation = np.array([[np.cos(angle),  -np.sin(angle)],
                           [np.sin(angle),  np.cos(angle)]])
    if matrix is None:
        return rotation

    return np.dot(matrix, rotation)


def create_random_rotation(angle_x = (0, 2*np.pi), angle_y = (0, 2*np.pi), angle_z = (0, 2*np.pi)):
    return create_matrix_rotation_x_3d(np.random.uniform(*angle_x), create_matrix_rotation_y_3d(np.random.uniform(*angle_y), create_matrix_rotation_z_3d(np.random.uniform(*angle_z))))

def convert_seg_image_to_one_hot_encoding(image):
    '''
    Takes as input an nd array of a label map (any dimension). Outputs a one hot encoding of the label map.
    Example (3D): if input is of shape (x, y, z), the output will ne of shape (n_classes, x, y, z)
    '''
    classes = np.unique(image)
    out_image = np.zeros([len(classes)]+list(image.shape), dtype=image.dtype)
    for i, c in enumerate(classes):
        out_image[i][image == c] = 1
    return out_image