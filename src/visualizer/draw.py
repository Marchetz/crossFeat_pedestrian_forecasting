from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import os
import matplotlib.patches as patches

def create_mapillary_vistas_label_colormap():
  """Creates a label colormap used in Mapillary Vistas segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([
      [165, 42, 42],
      [0, 192, 0],
      [196, 196, 196],
      [190, 153, 153],
      [180, 165, 180],
      [102, 102, 156],
      [64, 170, 64],
      [128, 64, 255],
      [140, 140, 200],
      [170, 170, 170],
      [250, 170, 160],
      [96, 96, 96],
      [230, 150, 140],
      [128, 64, 128],
      [110, 110, 110],
      [244, 35, 232],
      [150, 100, 100],
      [70, 70, 70],
      [150, 120, 90],
      [220, 20, 60],
      [255, 0, 0],
      [255, 0, 0],
      [255, 0, 0],
      [200, 128, 128],
      [255, 255, 255],
      [64, 170, 64],
      [128, 64, 64],
      [70, 130, 180],
      [255, 255, 255],
      [152, 251, 152],
      [107, 142, 35],
      [0, 170, 30],
      [255, 255, 128],
      [250, 0, 30],
      [0, 0, 0],
      [220, 220, 220],
      [170, 170, 170],
      [222, 40, 40],
      [100, 170, 30],
      [40, 40, 40],
      [33, 33, 33],
      [170, 170, 170],
      [0, 0, 142],
      [170, 170, 170],
      [210, 170, 100],
      [153, 153, 153],
      [128, 128, 128],
      [0, 0, 142],
      [250, 170, 30],
      [192, 192, 192],
      [220, 220, 0],
      [180, 165, 180],
      [119, 11, 32],
      [0, 0, 142],
      [0, 60, 100],
      [0, 0, 142],
      [0, 0, 90],
      [0, 0, 230],
      [0, 80, 100],
      [128, 64, 64],
      [0, 0, 110],
      [0, 0, 70],
      [0, 0, 192],
      [32, 32, 32],
      [0, 0, 0],
      [0, 0, 0],
      ])

def load_image(path, scale=1.0):
    with open(path, 'rb') as f:
        image = Image.open(f).convert('RGB')
        image = np.asarray(image) * scale / 255.0
        return image

    
def plot_image_tensor(img_tensor):
    Tensor2PIL = torchvision.transforms.ToPILImage(mode='RGB')
    pil_img = Tensor2PIL(img_tensor)
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    plt.imshow(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), aspect='equal')

def draw_ped_bbox(image, bbox):
    thickness = 2
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    color = (0, 255, 0)
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image
    
def draw_ped_ann(image, bbox, action=None):
    thickness = 2
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[2]), int(bbox[3]))
    if action:
        color = (0, 0, 255)
        org = (int(bbox[0] - 10), int(bbox[1] - 10))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = .75
        # image = cv2.putText(image, 'walking', org, font, fontScale, color, 2, cv2.LINE_AA)
    else:
        color = (0, 255, 0)
        org = (int(bbox[0]), int(bbox[1] - 10))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.75
        # image = cv2.putText(image, 'standing', org, font, fontScale, color, 2, cv2.LINE_AA)
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image

def draw_prediction(image, prediction):
    color = (56, 46, 237)
    org = (700, 100)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2.0
    #clone image
    temp = image.copy()
    image = cv2.putText(temp, 'model output: ' + str(prediction) + '%', org, font, fontScale, color, 2, cv2.LINE_AA)

    return image

def draw_frame_time(image, k=0, status=None):
    thickness = 30
    start_point = (0, 0)
    h = image.shape[0]
    w = image.shape[1]
    end_point = (w, h)
    if status == 'past':
        color = (164, 83, 0)
        org = (w - 240, 0 + 100)
        org_k = (0 + 40, 0 + 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 3.0
        image = cv2.putText(image, 'Past', org, font, fontScale, color, 2, cv2.LINE_AA)
        image = cv2.putText(image, 'Frame:' + str(k), org_k, font, fontScale, color, 2, cv2.LINE_AA)
    if status == 'future':
        color = (122, 80, 211)
        org = (w - 320, 0 + 100)
        org_k = (0 + 40, 0 + 100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 3.0
        image = cv2.putText(image, 'Future', org, font, fontScale, color, 2, cv2.LINE_AA)
        image = cv2.putText(image, 'Frame: ' + str(k), org_k, font, fontScale, color, 2, cv2.LINE_AA)
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    return image

def draw_banner(image):
    start_point = (20, 20)
    end_point = (1900, 150)
    color = (122, 80, 211)
    thickness = 2

    overlay = image.copy()

    x, y, w, h = 20, 20, 1880, 120  # Rectangle parameters
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (128, 128, 128), -1)  # A filled rectangle

    alpha = 0.4  # Transparency factor.

    # Following line overlays transparent rectangle over the image
    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image_new



class BaseVisualizer:
    def __init__(self, sample, bbox_value=None, nofuture=False):
        self.sample = sample
        self.source = sample['source']
        self.image_past = sample['qualitative_img_past']
        self.image_seg_sem = sample['img_sem_seg_qualitative']
        self.bbox_past = sample['bbox_past']
        self.len_past = self.image_past.shape[0]
        self.action_past = sample['action_past']
        if nofuture:
            self.image_future = sample['qualitative_img_future'].squeeze(0)
            self.len_future = self.image_future.shape[0]
            self.bbox_future = sample['bbox_future'].squeeze(0)
            self.images = torch.cat((self.image_past, self.image_future), dim=0)
            self.bboxes = torch.cat((self.bbox_past, self.bbox_future), dim=0)
            self.action_future = sample['action_future']
            self.action = torch.cat((self.action_past, self.action_future), dim=0).squeeze(0)
        else:
            self.images = self.image_past
            self.bboxes = self.bbox_past
            self.action = self.action_past

        self.label = sample['label'].int().item()
        self.tte = sample['tte'].item()
        self.image_path = sample['image_path']
        self.id = sample['id']
        self.bbox_value = bbox_value

        self.save_path = 'examples/' + self.id
        #create folder if not exist
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def show_sem_qualitative(self, time_flag=True, ped_flag=True, banner_flag=True, save_flag=False):
        k = 4
        status='past'

        bbox = self.bboxes[k]
        action = self.action[k]
        img = torch.Tensor(self.image_seg_sem)
        h = img.shape[0]
        w = img.shape[1]
        width = 1281
        height = 481
        if img.shape == torch.Size([1080, 1920]):
            cropping_ratio = 1/ 3
            image_new = img[int(h * cropping_ratio):h, 0:w]
            img = torch.Tensor(cv2.resize(image_new.numpy(), (width, height), interpolation=cv2.INTER_NEAREST))
        img = img.unsqueeze(2)
        #transorm image with one semantic channel in rgb
        cmap = torch.Tensor(create_mapillary_vistas_label_colormap())

        # / 255.0
        Tensor2PIL = torchvision.transforms.ToPILImage(mode='RGB')
        rgb_image = cmap[img.squeeze().long()]
        pil_img = Tensor2PIL(rgb_image.permute(2,0,1))
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_final = cv2_img.copy()

        if banner_flag:
            img_final = draw_banner(img_final)
        if ped_flag:
            img_final = draw_ped_ann(img_final, bbox, action)
        if time_flag:
            img_final = draw_frame_time(img_final, k, status)

        plt.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB), aspect='equal')

        if save_flag:
            #save figure in opencv
            #put zfill format
            cv2.imwrite(self.save_path + '/' + str(k).zfill(2) + '_SEM.png', img_final)
        else:
            plt.show()
        return cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)

    def show_frame_seg(self, k=0, prediction=None, time_flag=True, ped_flag=True, banner_flag=True, save_flag=False, title=None, bbox_flag=None):
        """
        Visualize kth frame in history sample
        """

        # bbox = self.bboxes[k]
        # action = self.action[k]
        h = self.image_seg_sem.shape[0]
        w = self.image_seg_sem.shape[1]
        width = 1281
        height = 481
        img = self.image_seg_sem
        if img.shape == torch.Size([1080, 1920, 3]):
            cropping_ratio = 1/ 3
            image_new = img[int(h * cropping_ratio):h, 0:w]
            img = torch.Tensor(cv2.resize(image_new.numpy(), (width, height), interpolation=cv2.INTER_NEAREST))


        Tensor2PIL = torchvision.transforms.ToPILImage(mode='RGB')
        pil_img = torch.Tensor(img)
        #pil_img = Tensor2PIL(img.permute(2,0,1))
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_final = cv2_img.copy()

        plt.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB), aspect='equal')

        if save_flag:
            cv2.imwrite(self.save_path + '/' + str(k).zfill(2) + '.png', img_final)
        else:
            plt.show()



    def show_frame(self, k=0, prediction=None, time_flag=True, ped_flag=True, banner_flag=True, save_flag=False, title=None, bbox_flag=None):
        """
        Visualize kth frame in history sample
        """
        if k < self.len_past:
            status='past'
        else:
            status='future'
        bbox = self.bboxes[k]
        action = self.action[k]
        h = self.images[k].shape[0]
        w = self.images[k].shape[1]
        width = 1281
        height = 481
        img = self.images[k] / 255.0
        if img.shape == torch.Size([1080, 1920, 3]):
            cropping_ratio = 1/ 3
            image_new = img[int(h * cropping_ratio):h, 0:w]
            img = torch.Tensor(cv2.resize(image_new.numpy(), (width, height), interpolation=cv2.INTER_NEAREST))

            if self.source == 'TITAN':
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
            bbox = bbox_new.copy()

        Tensor2PIL = torchvision.transforms.ToPILImage(mode='RGB')

        pil_img = Tensor2PIL(img.permute(2,0,1))
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img_final = cv2_img.copy()

        if banner_flag:
            img_final = draw_banner(img_final)
        if ped_flag:
            img_final = draw_ped_ann(img_final, bbox, action)
        if time_flag:
            img_final = draw_frame_time(img_final, k, status)
        if prediction is not None:
            img_final = draw_prediction(img_final, prediction)
        if bbox_flag is not None:
            img_final = draw_ped_bbox(img_final, bbox_flag)

        plt.imshow(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB), aspect='equal')

        if save_flag:
            cv2.imwrite(self.save_path + '/' + str(k).zfill(2) + '.png', img_final)
        else:
            plt.show()

    def show_history(self, prediction=None):
        for k in range(len(self.images)):
            if k<4:
                self.show_frame(k, prediction=None, save_flag=True)
            else:
                self.show_frame(k, prediction=prediction, save_flag=True)

            
