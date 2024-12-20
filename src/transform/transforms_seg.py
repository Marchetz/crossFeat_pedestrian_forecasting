import PIL
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import cv2

import matplotlib.patches as patches
def img_pad(img, mode='warp', size=224):
    '''
    Pads a given image.
    Crops and/or pads a image given the boundries of the box needed
    img: the image to be coropped and/or padded
    bbox: the bounding box dimensions for cropping
    size: the desired size of output
    mode: the type of padding or resizing. The modes are,
        warp: crops the bounding box and resize to the output size
        same: only crops the image
        pad_same: maintains the original size of the cropped box  and pads with zeros
        pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
        the desired output size in that direction while maintaining the aspect ratio. The rest of the image is
        padded with zeros
        pad_fit: maintains the original size of the cropped box unless the image is biger than the size in which case
        it scales the image down, and then pads it
    '''
    assert (mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
    image = img.copy()
    if mode == 'warp':
        warped_image = image.resize((size, size), PIL.Image.BICUBIC)
        return warped_image
    elif mode == 'same':
        return image
    elif mode in ['pad_same', 'pad_resize', 'pad_fit']:
        img_size = image.size  # size is in (width, height)
        ratio = float(size) / max(img_size)
        if mode == 'pad_resize' or \
                (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
            img_size = tuple([int(img_size[0] * ratio), int(img_size[1] * ratio)])
            image = image.resize(img_size, PIL.Image.BICUBIC)
        padded_image = PIL.Image.new("RGB", (size, size))
        padded_image.paste(image, ((size - img_size[0]) // 2,
                                   (size - img_size[1]) // 2))
        return padded_image


def squarify(bbox, squarify_ratio, img_width):
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height * squarify_ratio - width
    bbox[0] = bbox[0] - width_change / 2
    bbox[2] = bbox[2] + width_change / 2
    # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
    if bbox[0] < 0:
        bbox[0] = 0
    # check whether the new bounding box goes beyond image boarders
    # If this is the case, the bounding box is shifted back
    if bbox[2] > img_width:
        # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
        bbox[0] = bbox[0] - bbox[2] + img_width
        bbox[2] = img_width
    return bbox


def bbox_sanity_check(img, bbox):
    '''
    This is to confirm that the bounding boxes are within image boundaries.
    If this is not the case, modifications is applied.
    This is to deal with inconsistencies in the annotation tools
    '''
    img_width, img_heigth = img.shape[1], img.shape[0]
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= img_width:
        bbox[2] = img_width - 1
    if bbox[3] >= img_heigth:
        bbox[3] = img_heigth - 1
    return bbox


def jitter_bbox(img, bbox, mode, ratio):
    '''
    This method jitters the position or dimentions of the bounding box.
    mode: 'same' returns the bounding box unchanged
          'enlarge' increases the size of bounding box based on the given ratio.
          'random_enlarge' increases the size of bounding box by randomly sampling a value in [0,ratio)
          'move' moves the center of the bounding box in each direction based on the given ratio
          'random_move' moves the center of the bounding box in each direction by randomly sampling a value in [-ratio,ratio)
    ratio: The ratio of change relative to the size of the bounding box. For modes 'enlarge' and 'random_enlarge'
           the absolute value is considered.
    Note: Tha ratio of change in pixels is calculated according to the smaller dimension of the bounding box.
    '''
    assert (mode in ['same', 'enlarge', 'move', 'random_enlarge', 'random_move']), \
        'mode %s is invalid.' % mode

    if mode == 'same':
        return bbox

    if mode in ['random_enlarge', 'enlarge']:
        jitter_ratio = abs(ratio)
    else:
        jitter_ratio = ratio

    if mode == 'random_enlarge':
        jitter_ratio = np.random.random_sample() * jitter_ratio
    elif mode == 'random_move':
        # for ratio between (-jitter_ratio, jitter_ratio)
        # for sampling the formula is [a,b), b > a,
        # random_sample * (b-a) + a
        jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

    if len(bbox) == 4:
        b = copy.deepcopy(bbox)
        bbox_width = b[2] - b[0]
        bbox_height = b[3] - b[1]

        width_change = bbox_width * jitter_ratio
        height_change = bbox_height * jitter_ratio

        if width_change < height_change:
            height_change = width_change
        else:
            width_change = height_change

        if mode in ['enlarge', 'random_enlarge']:
            b[0] = b[0] - width_change // 2
            b[1] = b[1] - height_change // 2
        else:
            b[0] = b[0] + width_change // 2
            b[1] = b[1] + height_change // 2

        b[2] = b[2] + width_change // 2
        b[3] = b[3] + height_change // 2

        # Checks to make sure the bbox is not exiting the image boundaries
        jit_box = bbox_sanity_check(img, b)

    return jit_box

def draw(image, bb):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Create a Rectangle patch
    rect = patches.Rectangle((bb[0], bb[1]), (bb[2] - bb[0]), (bb[3] - bb[1]), linewidth=2, edgecolor='r',
                             facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

def crop_and_rescale_sem(image_or, image, bbox, bbox_history, pose, cropping_ratio, width, height, source):
    # crop the top 1/3 of image
    h, w = image.shape
    if image_or is not None:
        w_or = image_or.size[0]
        h_or = image_or.size[1]
        image_or_new = image_or.crop((0, h_or * cropping_ratio, w_or, h_or))
        image_or_new_res = image_or_new.resize((width, height), PIL.Image.BICUBIC)
    else:
        image_or_new_res = image_or

    #sem
    if image.shape != (481,1281):
        #h, w = image.shape
        image_new = image[int(h * cropping_ratio):h, 0:w]
        image_new_res = cv2.resize(image_new, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        image_new_res = image

    #correct
    if source == 'TITAN':
        h = 1520
        w = 2704
    else:
        h = 1080
        w = 1920
    scale_x = width / w
    scale_y = height / h / (1 - cropping_ratio)
    x1 = bbox[0] * scale_x
    y1 = (bbox[1] - h * cropping_ratio) * scale_y
    x2 = bbox[2] * scale_x
    y2 = (bbox[3] - h * cropping_ratio) * scale_y
    bbox_new = [int(x1), int(y1), int(x2), int(y2)]
    bbox_ped_new = copy.deepcopy(bbox_new)

    #correct
    if bbox_history is not None:
        for i in range(len(bbox_history)):
            x1 = bbox_history[i][0] * scale_x
            y1 = (bbox_history[i][1] - h * cropping_ratio) * scale_y
            x2 = bbox_history[i][2] * scale_x
            y2 = (bbox_history[i][3] - h * cropping_ratio) * scale_y
            bbox_history[i] = [int(x1), int(y1), int(x2), int(y2)]

    # compute new pose
    if pose is not None:
        for i in range(len(pose)):

            pose[i,:,0] = pose[i,:,0] * scale_x
            pose[i,:,1] = (pose[i,:,1] - h * cropping_ratio) * scale_y
    else:
        pose = None

    #return image_or_new_res, image_new_res, depth_new_res, bbox_new, bbox_ped_new, bbox_history, pose
    return image_or_new_res, image_new_res, bbox_new, bbox_ped_new, bbox_history, pose

def crop_and_rescale(image, bbox, cropping_ratio, width, height):
    """
    Crop the top 1/n of the image and resize the image to desired size.
    The bbox are preprocessed accordingly.
    """
    w, h = image.size
    image = image.crop((0, h * cropping_ratio, w, h))
    # rescale
    image_new = image.resize((width, height), PIL.Image.BICUBIC)
    # compute new bbox
    scale_x = width / w
    scale_y = height / h / (1 - cropping_ratio)
    x1 = bbox[0] * scale_x
    y1 = (bbox[1] - h * cropping_ratio) * scale_y
    x2 = bbox[2] * scale_x
    y2 = (bbox[3] - h * cropping_ratio) * scale_y
    bbox_new = [int(x1), int(y1), int(x2), int(y2)]

    return image_new, bbox_new



def random_flip(image, bbox, probability):
    if float(torch.rand(1).item()) < probability:
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        w, h = image.size
        # box_w = abs(bbox[0] - bbox[2])
        x_max = w - bbox[0]
        x_min = w - bbox[2]
        bbox[0] = x_min
        bbox[2] = x_max

    return image, bbox
