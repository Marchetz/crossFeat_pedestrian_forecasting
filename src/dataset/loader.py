import logging
import sys
import math
LOG = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from src.dataset.trans.data import *
from src.transform.preprocess_seg import *
from src.visualizer.draw import *

classes = ['Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail', 'Barrier', 'Wall', 'Bike Lane',
           'Crosswalk - Plain', 'Curb Cut', 'Parking', 'Pedestrian Area', 'Rail Track', 'Road', 'Service Lane',
           'Sidewalk', 'Bridge', 'Building', 'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider',
           'Lane Marking - Crosswalk', 'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain',
           'Vegetation', 'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
           'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole', 'Phone Booth', 'Pothole', 'Street Light',
           'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light', 'Traffic Sign (Back)',
           'Traffic Sign (Front)', 'Trash Can', 'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle',
           'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow', 'Car Mount', 'Ego Vehicle']
classes_label = torch.arange(len(classes)).tolist()

dict_group_classes = {
    'Road': ['Road', 'Lane Marking - General', 'Manhole'],
    'Vegetation': ['Vegetation', 'Terrain', 'Sky', 'Mountain', 'Sand', 'Snow', 'Water'],
    'Buildings': ['Building', 'Wall', 'Fence', 'Guard Rail', 'Bridge', 'Tunnel', 'Billboard', 'Banner', 'Rail Track',
                  'Barrier'],
    'Ego': ['Ego Vehicle', 'Car Mount'],
    'Car': ['Car', 'Truck', 'Bus', 'Trailer', 'Boat', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle'],
    'Pedestrian': ['Person', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'Bicycle', 'Wheeled Slow'],
    'Sidewalk': ['Sidewalk', 'Parking', 'Pedestrian Area', 'Curb Cut', 'Curb'],
    'Traffic': ['Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Traffic Sign Frame'],
    'Crosswalk': ['Crosswalk - Plain', 'Lane Marking - Crosswalk', 'Service Lane', 'Bike Lane'],
    'Others': ['Bench', 'Bike Rack', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box', 'Mailbox',
               'Phone Booth', 'Pothole', 'Street Light', 'Pole', 'Trash Can', 'Bird', 'Ground Animal', 'Utility Pole']
}

dict_new_label = {'Road': 0, 'Vegetation': 1, 'Buildings': 2, 'Ego': 3, 'Car': 4,
                  'Pedestrian': 5, 'Sidewalk': 6, 'Traffic': 7, 'Crosswalk': 8,
                  'Others': 9}
classes_10 = ['Road', 'Vegetation', 'Buildings', 'Ego', 'Car', 'Pedestrian', 'Sidewalk', 'Traffic', 'Crosswalk',
                'Others']


def define_path(use_jaad=True, use_pie=True, use_titan=True, use_loki=False):
    """
    Define default path to data
    """
    # main_path = '/equilibrium/datasets/TransNet/DATA/'
    main_path = '/seidenas/users/fmarchetti/Datasets/TransNet/DATA/'

    main_path_PIE = '/seidenas/users/fmarchetti/Datasets/TransNet/PIE/'
    all_anns_paths = {'JAAD': {'anns': main_path + 'annotations/JAAD/anns/JAAD_DATA.pkl',
                               'split': main_path + 'annotations/JAAD/splits/'},
                      'PIE': {'anns': f'{main_path_PIE}/data_cache/pie_database.pkl'},
                      'TITAN': {'anns': main_path + 'annotations/TITAN/anns/',
                                'split': main_path + 'annotations/TITAN/splits/'},
                      'LOKI': {'anns': '/equilibrium/fmarchetti/LOKI'},

                      }
    all_image_dir = {'JAAD': main_path + 'images/JAAD/',
                     'PIE': f'{main_path_PIE}/images/',
                     'TITAN': main_path + 'images/TITAN/images_anonymized/',
                     'LOKI': '/equilibrium/fmarchetti/LOKI'
                     }

    anns_paths = {}
    image_dir = {}
    if use_jaad:
        anns_paths['JAAD'] = all_anns_paths['JAAD']
        image_dir['JAAD'] = all_image_dir['JAAD']
    if use_pie:
        anns_paths['PIE'] = all_anns_paths['PIE']
        image_dir['PIE'] = all_image_dir['PIE']
    if use_titan:
        anns_paths['TITAN'] = all_anns_paths['TITAN']
        image_dir['TITAN'] = all_image_dir['TITAN']
    if use_loki:
        anns_paths['LOKI'] = all_anns_paths['LOKI']
        image_dir['LOKI'] = all_image_dir['LOKI']

    return anns_paths, image_dir


class PaddedSequenceDataset_segSem(torch.utils.data.Dataset):
    """
    Basic dataloader for loading sequence/history samples
    """

    def __init__(self, type_dataset, samples, image_dir, padded_length=10, preprocess=None, hflip_p=0.0, debug=False, qualitative=False, depth=None, args=None, preprocess_1=None, preprocess_2=None):
        """
        :params: samples: transition history samples(dict)
                image_dir: root dir for images extracted from video clips
                preprocess: optional preprocessing on image tensors and annotations
        """
        self.type_dataset = type_dataset
        self.samples = samples
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.preprocess_1 = preprocess_1
        self.preprocess_2 = preprocess_2
        self.padded_length = padded_length
        self.hflip_p = hflip_p
        self.debug = debug
        self.qualitative = qualitative
        self.depth_dict = depth
        self.args = args
        if depth is not None:
            self.depth = True

        self.path_image = []
        self.type_seg = 'semantic'
        self.bbox_generated = 'classic'


    def get_group(self, class_name):
        for key in dict_group_classes:
            if class_name in dict_group_classes[key]:
                return key

    def convert_image(self, image):
        image = torch.tensor(image)  # convert image to a tensor
        image_new = torch.zeros_like(image)
        for i, class_label in enumerate(classes_label):
            image_new[image == class_label] = dict_new_label[self.get_group(classes[i])]
        assert (image_new.unique().int() > 9).sum() == 0
        return image_new  # .tolist() # convert tensor back to a list

    def draw(self, image, bb=None, pose=None, title='image'):
        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(image)

        # Create a Rectangle patch
        if bb is not None:
            rect = patches.Rectangle((bb[0], bb[1]), (bb[2] - bb[0]), (bb[3] - bb[1]), linewidth=2, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
        if pose is not None:
            for i in range(len(pose)):
                ax.plot(pose[i][0], pose[i][1], 'b.')
        plt.title(title)
        plt.show()

    def get_sample(self, id):
        data = self.__getitem__(0, choose_sample=id)
        return data

    def get_qualitative_data(self, image_path_qualitative, frames, frames_future, bbox, bbox_future, hflip, w_or, source, action, action_future):
        qualitative_img_past = []
        qualitative_img_future = []
        bbox_past = []
        bbox_future_temp = []
        action_past = action

        # get qualitative data for past
        for step, j in enumerate(frames):
            if source == "JAAD":
                img_qual_path = image_path_qualitative + '/{:05d}.png'.format(j)
            elif source == "PIE":
                img_qual_path = image_path_qualitative + '/{:05d}.jpg'.format(j)
            elif source == "TITAN":
                img_qual_path = image_path_qualitative + '/{:06}.png'.format(j)
            bbox_qual = copy.deepcopy(bbox[step])
            with open(img_qual_path, 'rb') as f:
                image_qual = PIL.Image.open(f).convert('RGB')
            if hflip:
                image_qual = image_qual.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                x_max = w_or - bbox_qual[0]
                x_min = w_or - bbox_qual[2]
                bbox_qual[0] = x_min
                bbox_qual[2] = x_max

            qualitative_img_past.append(torch.Tensor(np.array(image_qual)).unsqueeze(0))
            bbox_past.append(bbox_qual)


        for step, j in enumerate(frames_future):
            if source == "JAAD":
                img_qual_path = image_path_qualitative + '/{:05d}.png'.format(j)
            elif source == "PIE":
                img_qual_path = image_path_qualitative + '/{:05d}.jpg'.format(j)
            elif source == "TITAN":
                img_qual_path = image_path_qualitative + '/{:06}.png'.format(j)

            bbox_qual = copy.deepcopy(bbox_future[step])


            with open(img_qual_path, 'rb') as f:
                image_qual = PIL.Image.open(f).convert('RGB')
            if hflip:
                image_qual = image_qual.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                x_max = w_or - bbox_qual[0]
                x_min = w_or - bbox_qual[2]
                bbox_qual[0] = x_min
                bbox_qual[2] = x_max
            qualitative_img_future.append(torch.Tensor(np.array(image_qual)).unsqueeze(0))
            bbox_future_temp.append(bbox_qual)
        return qualitative_img_past, qualitative_img_future, bbox_past, bbox_future_temp, action_past, action_future

    def __getitem__(self, index, choose_sample=False):
        if choose_sample:
            idx = choose_sample
        else:
            ids = list(self.samples.keys())
            idx = ids[index]
        frames = self.samples[idx]['frame']
        bbox = copy.deepcopy(self.samples[idx]['bbox'])
        source = self.samples[idx]["source"]
        action = self.samples[idx]['action']
        speed = self.samples[idx]['speed']

        frames_future = self.samples[idx]['frame_future']
        bbox_future = copy.deepcopy(self.samples[idx]['bbox_future'])
        action_future = self.samples[idx]['action_future']
        TTE = self.samples[idx]["TTE"]

        if 'trans_label' in list(self.samples[idx].keys()):
            label = self.samples[idx]['trans_label']
        else:
            label = None
        if 'behavior' in list(self.samples[idx].keys()):
            behavior = self.samples[idx]['behavior']
        else:
            behavior = [-1, -1, -1, -1]
        if 'attributes' in list(self.samples[idx].keys()):
            attributes = self.samples[idx]['attributes']
        else:
            attributes = [-1, -1, -1, -1, -1, -1]
        """
        if 'traffic_light' in list(self.samples[idx].keys()):
            traffic_light = self.samples[idx]['traffic_light']
        else:
            traffic_light = []
        """
        bbox_frame = []
        image_path = None
        hflip = True if float(torch.rand(1).item()) < self.hflip_p else False
        if source == 'TITAN':
            w_or = 2704
        else:
            w_or = 1920

        i = -1
        #suffix = 'CS_R50'
        suffix = ''
        anns = {'bbox_history': bbox, 'bbox': bbox[i].copy(), 'source': source}
        if source == "JAAD":
            vid = self.samples[idx]['video_number']
            image_path_qualitative = os.path.join(self.image_dir['JAAD'], vid)
            #image_path = os.path.join(self.image_dir['JAAD'], vid, '{:05d}_seg_sem.png'.format(frames[i]))
            image_path = os.path.join(self.image_dir['JAAD'], vid, '{:05d}_seg_sem{}.png'.format(frames[i], suffix))
            image_path_or = os.path.join(self.image_dir['JAAD'], vid, '{:05d}.png'.format(frames[i]))

        elif source == "PIE":
            vid = self.samples[idx]['video_number']
            sid = self.samples[idx]['set_number']
            image_path_qualitative = os.path.join(self.image_dir['PIE'], sid, vid)
            #image_path = os.path.join(self.image_dir['PIE'], sid, vid, '{:05d}_seg_sem.png'.format(frames[i]))
            image_path = os.path.join(self.image_dir['PIE'], sid, vid, '{:05d}_seg_sem{}.png'.format(frames[i], suffix))
            image_path_or = os.path.join(self.image_dir['PIE'], sid, vid, '{:05d}.jpg'.format(frames[i]))
        elif source == "TITAN":
            vid = self.samples[idx]['video_number']
            image_path_qualitative = os.path.join(self.image_dir['TITAN'], vid, 'images')
            #image_path = os.path.join(self.image_dir['TITAN'], vid, 'images', '{:06}_seg_sem.png'.format(frames[i]))
            image_path = os.path.join(self.image_dir['TITAN'], vid, 'images', '{:06}_seg_sem{}.png'.format(frames[i], suffix))
            image_path_or = os.path.join(self.image_dir['TITAN'], vid, 'images', '{:06}.png'.format(frames[i]))

        if self.qualitative:
            qualitative_img_past, qualitative_img_future, bbox_past, bbox_future, action_past, action_future = \
                self.get_qualitative_data(image_path_qualitative, frames, frames_future, bbox, bbox_future,
                                          hflip, w_or, source, action, action_future)

        #load rgb image
        if self.debug:
            with open(image_path_or, 'rb') as f:
                img_or = PIL.Image.open(f).convert('RGB')
        else:
            img_or = None

        # load semantic segmentation image
        with open(image_path, 'rb') as f:
            img = imageio.imread(image_path)

        #check image size
        if img.shape == (512,512):
            print('error 512 before')
            print('img shape: ', img.shape)
            print('path: ', image_path)
        size_original = img.shape
        self.path_image.append(image_path)
        img = np.array(img)
        img_sem = img.copy()

        # if self.debug:
        #     img_before_or = img_or.copy()
        #     img_before = img_sem.copy()
        #     bbox_before = anns['bbox'].copy()

        #check draw data:
        # file_path_or = '/equilibrium/datasets/TransNet/DATA/images/JAAD/video_0020/00363.png'
        # file_path = '/equilibrium/datasets/TransNet/DATA/images/JAAD/video_0020/00363_seg_sem.png'
        #
        # #open file path
        # with open(file_path_or, 'rb') as f:
        #     img_or = PIL.Image.open(f).convert('RGB')
        # with open(file_path, 'rb') as f:
        #     img = imageio.imread(file_path)
        # #save img with opencv with colormap
        # #convert img to colormap rgb
        # img = np.array(img)
        # img_sem = img.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        #cv2.imwrite('img_2.png', img)
        current_pose = None
        # self.draw(img, bbox_before, current_pose, title="before")
        # self.draw(img_before, bbox_before, current_pose, title="before")
        # self.draw(img_before_or, bbox_before, current_pose, title="before")

        #flip augmentation
        if hflip:
            img = np.fliplr(img)
            img_sem = np.fliplr(img_sem)
            # if depth is not None:
            #     depth = np.fliplr(depth)
            if img_or is not None:
                img_or = img_or.transpose(PIL.Image.FLIP_LEFT_RIGHT)

            temp = anns['bbox'].copy()
            x_max = w_or - anns['bbox'][0]
            x_min = w_or - anns['bbox'][2]
            anns['bbox'][0] = x_min
            anns['bbox'][2] = x_max

            for i in range(len(anns['bbox_history'])):
                x_max = w_or - anns['bbox_history'][i][0]
                x_min = w_or - anns['bbox_history'][i][2]
                anns['bbox_history'][i][0] = x_min
                anns['bbox_history'][i][2] = x_max
        bbox_original = anns['bbox'].copy()
        img_sem_seg_qualitative = img_sem.copy()
        ########################################################
        if self.bbox_generated == 'classic':
            if self.preprocess is not None:
                img_or, img, anns = self.preprocess(img_or, img, anns)
            image_path_total = []
        else:
            img_or, img, anns = self.preprocess_1(img_or, img, anns)

            image_path_total = []
            # self.draw(img_pan, anns['bbox_history'][0], title='after')
            bbox_total_ex = []
            for i in range(len(frames)):
                if self.debug:
                    i = -1
                if source == "JAAD":
                    vid = self.samples[idx]['video_number']
                    image_path_new = os.path.join(self.image_dir['JAAD'], vid, '{:05d}.png'.format(frames[i]))
                    image_path_pan = os.path.join(self.image_dir['JAAD'], vid, '{:05d}_pan_r50.png'.format(frames[i]))
                    segment_pan = os.path.join(self.image_dir['JAAD'], vid, '{:05d}_pan_r50_text.pkl'.format(frames[i]))
                elif source == "PIE":
                    vid = self.samples[idx]['video_number']
                    sid = self.samples[idx]['set_number']
                    image_path_new = os.path.join(self.image_dir['PIE'], sid, vid, '{:05d}.jpg'.format(frames[i]))
                    image_path_pan = os.path.join(self.image_dir['PIE'], sid, vid, '{:05d}_pan_r50.png'.format(frames[i]))
                    segment_pan = os.path.join(self.image_dir['PIE'], sid, vid, '{:05d}_pan_r50_text.pkl'.format(frames[i]))
                elif source == "TITAN":
                    vid = self.samples[idx]['video_number']
                    image_path_new = os.path.join(self.image_dir['TITAN'], vid, 'images', '{:06}.png'.format(frames[i]))
                    image_path_pan = os.path.join(self.image_dir['TITAN'], vid, 'images', '{:06d}_pan_r50.png'.format(frames[i]))
                    segment_pan = os.path.join(self.image_dir['TITAN'], vid, 'images', '{:06d}_pan_r50_text.pkl'.format(frames[i]))
                image_path_total.append(image_path_new)
                bbox_gt = anns['bbox_history'][i]

                with open(image_path, 'rb') as f:
                    img_pan = imageio.imread(image_path_pan)
                    #open pkl
                with open(segment_pan, 'rb') as f:
                    segment = pickle.load(f)

                segment_people = []
                id_people = []
                for s in segment:
                    if s['category_id'] == 19:
                        segment_people.append(s)
                        id_people.append(s['id'])

                flag_segmentation = self.type_seg
                if flag_segmentation == 'panoptic':
                    if len(segment_people) == 0:
                        flag_segmentation = 'semantic'
                    else:
                        flag_segmentation = 'panoptic'


                if flag_segmentation == 'semantic':
                    mask_total = []
                    temp = img == 19
                    mask_total.append(temp)
                else:
                    mask_total = []
                    for s in segment_people:
                        temp = img_pan == s['id']
                        mask_total.append(temp)

                bbox_total = []
                for m in mask_total:
                    m = m.astype(np.uint8)
                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        bbox = [x, y, x+w, y+h]
                        bbox_total.append(bbox)

                try:
                    most_similar_bbox = self.find_most_similar_bbox(bbox_gt, bbox_total)
                except:
                    most_similar_bbox = [0, 0, 0, 0]
                bbox_total_ex.append(most_similar_bbox)
                # self.draw(img_pan, bbox_gt, title='gt')
                # self.draw(img_pan, bbox_total[0], title='after')
                # self.draw(img_pan, most_similar_bbox, title='generated')

            if self.bbox_generated == 'present':
                anns['bbox'] = bbox_total_ex[-1]
                anns['bbox_cb'] = bbox_total_ex[-1]
                # anns['bbox_history'][-1] = bbox_total_ex[-1]
            if self.bbox_generated == 'present_past':
                anns['bbox'] = bbox_total_ex[-1]
                anns['bbox_cb'] = bbox_total_ex[-1]
                anns['bbox_history'] = bbox_total_ex

            img_or, img, anns = self.preprocess_2(img_or, img, anns)

            while len(image_path_total) < 5:
                image_path_total.append(image_path_total[-1])
        #####################################################################



        if source == "JAAD" or source == "PIE":
            check_img = (1080, 1920)
        else:
            check_img = (1520, 2704)
        if img_sem.shape != check_img:
            img = img_sem.copy()

        if img_or is not None:
            img_or = np.array(img_or)

        #insert here the new bounding box


        #bbox in image
        bbox_frame.append(anns['bbox'])

        #normalized bbox
        bbox_norm = anns['bbox_cb'].copy()
        bbox_norm[0] = bbox_norm[0] / img.shape[1]
        bbox_norm[1] = bbox_norm[1] / img.shape[0]
        bbox_norm[2] = bbox_norm[2] / img.shape[1]
        bbox_norm[3] = bbox_norm[3] / img.shape[0]

        # sequence of bbox, action, behavior, traffic light
        bbox_history = anns['bbox_history']
        seq_len = len(bbox_history)
        bbox_new_padded = copy.deepcopy(bbox_history)
        action_padded = copy.deepcopy(action)
        behavior_padded = copy.deepcopy(behavior)
        speed_padded = copy.deepcopy(speed)
        # traffic_light_padded = copy.deepcopy(traffic_light)
        for i in range(len(bbox_new_padded), self.padded_length):
            bbox_new_padded.append([0, 0, 0, 0])
            action_padded.append(-1)
            speed_padded.append(-1)
            behavior_padded.append([-1, -1, -1, -1])
            # traffic_light_padded.append(-1)
        bbox_history = bbox_new_padded

        #normalized bbox history
        bbox_history_norm = copy.deepcopy(bbox_history)
        # todo: commentare?!?
        for i in range(len(bbox_history_norm)):
            bbox_history_norm[i][0] = bbox_history_norm[i][0] / img.shape[1]
            bbox_history_norm[i][1] = bbox_history_norm[i][1] / img.shape[0]
            bbox_history_norm[i][2] = bbox_history_norm[i][2] / img.shape[1]
            bbox_history_norm[i][3] = bbox_history_norm[i][3] / img.shape[0]

        if label is not None:
            label = torch.tensor(label)
            label = label.to(torch.float32)
        TTE_tag = -1
        if math.isnan(TTE):
            TTE = 0.0
        else:
            TTE = round(self.samples[idx]["TTE"], 2)
            if TTE < 0.45:
                TTE_tag = 0
            elif 0.45 < TTE < 0.85:
                TTE_tag = 1
            elif 0.85 < TTE < 1.25:
                TTE_tag = 2
            elif 1.25 < TTE < 1.65:
                TTE_tag = 3
            elif 1.65 < TTE < 2.05:
                TTE_tag = 4
            else:
                TTE_tag = 5
        TTE = torch.tensor(TTE).to(torch.float32)
        TTE_tag = torch.tensor(TTE_tag)
        TTE_tag = TTE_tag.to(torch.float32)
        attributes = torch.tensor(attributes).to(torch.float32)

        #debug draw
        # self.draw(img_or, anns['bbox_cb'], title='after')
        # self.draw(img, anns['bbox_cb'], title='after')  #bbox original
        # self.draw(img, anns['bbox'], title='after')  #with padding

        # todo
        if self.debug:
            img_all_sem = img.copy()
        else:
            img_all_sem = 0
        img = self.convert_image(img)

        #get area of bbox_frame
        width = anns['bbox_cb'][2] - anns['bbox_cb'][0]
        height = anns['bbox_cb'][3] - anns['bbox_cb'][1]
        area = ((width * height) / (img.shape[1] * img.shape[0])) * 100

        if img.shape != (481, 1281):
            # dtype is int
            img = torch.zeros((481,1281), dtype=torch.int)
        if img_or is None:
            img_or = 0
        if self.qualitative:
            qualitative_img_past = torch.cat(qualitative_img_past).int()
            qualitative_img_future = torch.cat(qualitative_img_future).int()
            bbox_past = torch.Tensor(bbox_past).int()
            bbox_future = torch.Tensor(bbox_future).int()
            action_past = torch.Tensor(action_past).int()
            action_future = torch.Tensor(action_future).int()
            len_future_qualitative = len(action_future)
        else:
            qualitative_img_past = 0
            qualitative_img_future = 0
            bbox_past = 0
            bbox_future = 0
            action_past = 0
            action_future = 0
            img_sem_seg_qualitative = 0
            len_future_qualitative = 0


        sample = {'image': img, 'img_all_sem': img_all_sem, 'img_or': img_or,
                  'bbox_norm': bbox_norm, 'bbox_ped': bbox_frame, 'bbox_history': bbox_history, 'bbox_cb': anns['bbox_cb'],
                  'bbox_history_norm': bbox_history_norm, 'bbox_original': bbox_original,
                  'seq_length': seq_len, 'action': action_padded, 'id': idx, 'label': label,
                  'source': source, 'tte': TTE, 'TTE_tag': TTE_tag,
                  'behavior': behavior_padded, 'attributes': attributes, 'image_path': image_path,
                  'size_original': size_original, 'area': area, 'qualitative_img_past': qualitative_img_past,
                    'qualitative_img_future': qualitative_img_future, 'bbox_past': bbox_past, 'bbox_future': bbox_future,
                    'action_past': action_past, 'action_future': action_future,
                  'img_sem_seg_qualitative': img_sem_seg_qualitative, 'len_future_qualitative': len_future_qualitative,
                  'speed': speed_padded, 'image_path_total': image_path_total
                  }

        return sample


    def calculate_iou(self, box1, box2):
        # Calculate the intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate the area of intersection
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate the area of both bounding boxes
        area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Calculate the IoU
        iou = intersection_area / float(area_box1 + area_box2 - intersection_area)

        return iou

    def create_bounding_box(self, binary_image):
        # Find the indices where the value is 1
        indices = np.where(binary_image == 1)

        # Get the minimum and maximum values for x and y
        min_x, min_y = np.min(indices[1]), np.min(indices[0])
        max_x, max_y = np.max(indices[1]), np.max(indices[0])

        # Create a bounding box
        bounding_box = (min_x, min_y, max_x, max_y)

        return bounding_box

    def __len__(self):
        return len(self.samples.keys())





