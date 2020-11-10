"""
Misc Utility functions
"""
import os
import logging
import datetime
import numpy as np

from collections import OrderedDict, deque


road_nroad_colors = np.array([(0, 255, 255), (70, 130, 180), (255, 255, 0)])

catId_colors = np.array([
    (0, 255, 255),  # road
    (244, 35, 232),  # sidewalk
    (150, 80, 80),  # building
    # (70, 70, 70),  # building
    (250, 170, 30),  # pole, traffic light, traffic sign
    (107, 142, 35),  # vegetation, terrain
    (70, 130, 180),  # sky
    (220, 20, 60),  # person, rider
    (0, 0, 142),  # car, truck, bus, train, motorcycle, bicycle
    (0, 0, 0)  # boundary
])

catId_colors_20 = np.array([
    # (0,0,0),
    (0, 0, 0),
    (0, 255, 255),
    (244, 35, 232),
    (150, 80, 80),
    (102, 102, 156),
    (190, 153, 153),
    (250, 170, 30),
    (250, 170, 30),
    (250, 170, 30),
    (107, 142, 35),
    (107, 142, 35),
    (70, 130, 180),
    (220, 20, 60),
    (220, 20, 60),
    (0, 0, 142),
    (0, 0, 142),
    (0, 0, 142),
    (0, 0, 142),
    (0, 0, 142),
    (0, 0, 142)
])

catId_colors_ori_20 = np.array([
    (0,0,0),
    (0, 0, 0),
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32)
])

color_encoding = OrderedDict([
    ('unlabeled', (0, 0, 0)),
    ('road', (128, 64, 128)),
    ('sidewalk', (244, 35, 232)),
    ('building', (70, 70, 70)),
    ('wall', (102, 102, 156)),
    ('fence', (190, 153, 153)),
    ('pole', (153, 153, 153)),
    ('traffic_light', (250, 170, 30)),
    ('traffic_sign', (220, 220, 0)),
    ('vegetation', (107, 142, 35)),
    ('terrain', (152, 251, 152)),
    ('sky', (70, 130, 180)),
    ('person', (220, 20, 60)),
    ('rider', (255, 0, 0)),
    ('car', (0, 0, 142)),
    ('truck', (0, 0, 70)),
    ('bus', (0, 60, 100)),
    ('train', (0, 80, 100)),
    ('motorcycle', (0, 0, 230)),
    ('bicycle', (119, 11, 32))
])

# color_encoding = OrderedDict([
#         ('unlabeled', (0, 0, 0)),
#         ('road', (0, 255, 255)),
#         ('sidewalk', (244, 35, 232)),
#         ('building', (150, 80, 80)),
#         ('wall', (102, 102, 156)),
#         ('fence', (190, 153, 153)),
#         ('pole', (250, 170, 30)),
#         ('traffic_light', (250, 170, 30)),
#         ('traffic_sign', (250, 170, 30)),
#         ('vegetation', (107, 142, 35)),
#         ('terrain', (107, 142, 35)),
#         ('sky', (70, 130, 180)),
#         ('person', (220, 20, 60)),
#         ('rider', (220, 20, 60)),
#         ('car', (0, 0, 142)),
#         ('truck', (0, 0, 142)),
#         ('bus', (0, 0, 142)),
#         ('train', (0, 0, 142)),
#         ('motorcycle', (0, 0, 142)),
#         ('bicycle', (0, 0, 142))
#     ])

a2d2_colors = np.array([  # RGB
    (0, 255, 255),  # road
    (244, 35, 232),  # curb
    (250, 170, 30),  # block
    (107, 142, 35),  # guidance
    (0, 0, 142),  # vehicle
    (142, 100, 0),  # bicycle
    (220, 20, 60),  # pedestrian
    (70, 130, 180),  # notRoad
    (0, 0, 0)  # boundary
])

color_codes = catId_colors_ori_20
print(color_codes)


def create_full_color_mask(mask, color_codes):
    """
    create color mask for all the objects
    :param mask:
    :return:
    """
    color_mask = color_codes[mask]
    return color_mask

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            name = k.replace('module.', '')  # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def seed_filling(mask, seed_coord, seed_value, connected_value):
    """
    find the maximum connected area starting from the seed pixel in the mask
    Here, the 4-neighbor relationship is used. The searching order is top-right-bottom-left
    :param mask: a single channel feature map with values indicating different classes
    :param seed_coord: [height, width] from which the connected pixels are searched
    :param seed_value: the value to be searched around the seed_coord
    :param connected_value: the value assigned to the connected component. Note that the connected value
    should be different from the seed_value
    :return: a binary mask with value 1 indicating the max connected area with the giving seed pixel in it
    """
    if mask[seed_coord] != seed_value:
        return mask
    mask_h, mask_w = mask.shape
    stack = deque()
    stack.append(seed_coord)
    while stack:
        current_pixel = stack.pop()
        h, w = current_pixel
        mask[current_pixel] = connected_value
        # top
        if h-1 >= 0 and mask[h-1, w] == seed_value:
            stack.append((h-1, w))
        # right
        if w+1 < mask_w and mask[h, w+1] == seed_value:
            stack.append((h, w+1))
        # bottom
        if h+1 < mask_h and mask[h+1, w] == seed_value:
            stack.append((h+1, w))
        # left
        if w-1 >= 0 and mask[h, w-1] == seed_value:
            stack.append((h, w-1))

    return mask


def extract_bottom_boundary(mask, pad_num, pad_value):
    """
    To extract the bottom boundary, the top of the mask is padded.
    :param mask: from which the boundary is extracted
    :param pad_num: number of pads to be added
    :param pad_value: value of the pads
    :return: boundary mask with boolean value type
    """
    mask_pad = np.ones_like(mask) * pad_value
    mask_pad[pad_num:] = mask[:-pad_num]
    mask_boundary = mask_pad != mask

    return mask_boundary
