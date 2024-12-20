import torch

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
'Road' : ['Road', 'Lane Marking - General', 'Manhole'],
'Vegetation' : ['Vegetation', 'Terrain', 'Sky', 'Mountain', 'Sand', 'Snow', 'Water'],
'Buildings' : ['Building', 'Wall', 'Fence', 'Guard Rail', 'Bridge', 'Tunnel', 'Billboard', 'Banner', 'Rail Track', 'Barrier'],
'Ego' : ['Ego Vehicle', 'Car Mount'],
'Car' : ['Car', 'Truck', 'Bus', 'Trailer', 'Boat', 'Caravan', 'Motorcycle', 'On Rails', 'Other Vehicle'],
'Pedestrian' : ['Person', 'Bicyclist', 'Motorcyclist', 'Other Rider', 'Bicycle', 'Wheeled Slow'],
'Sidewalk' : ['Sidewalk', 'Parking', 'Pedestrian Area', 'Curb Cut', 'Curb'],
'Traffic' : ['Traffic Light', 'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Traffic Sign Frame'],
'Crosswalk' : ['Crosswalk - Plain', 'Lane Marking - Crosswalk', 'Service Lane', 'Bike Lane'],
'Others' : ['Bench', 'Bike Rack', 'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box', 'Mailbox',
        'Phone Booth', 'Pothole', 'Street Light', 'Pole', 'Trash Can', 'Bird', 'Ground Animal', 'Utility Pole']
}

dict_new_label = {'Road': 0, 'Vegetation': 1, 'Buildings': 2, 'Ego': 3, 'Car': 4,
                    'Pedestrian': 5, 'Sidewalk': 6, 'Traffic': 7, 'Crosswalk': 8,
                    'Others': 9}

examples_count = {
    'JAAD': {
        'GO': {
            'pos': 503,
            'neg': 764
        },
        'STOP': {
            'pos': 179,
            'neg': 5574
        }
    },
    'PIE': {
        'GO': {
                'pos': 1287,
                'neg': 10871

        },
        'STOP': {
                'pos': 1524,
                'neg': 13586
        }
    },
    'TITAN': {
        'GO': {
            'pos': 1204,
            'neg': 5900

        },
        'STOP': {
            'pos': 1560,
            'neg': 8369
        }
    }
}

def convert_batch(batch):
    batch = torch.tensor(batch) # convert batch to a tensor
    batch_new = torch.zeros_like(batch)
    for i, class_label in enumerate(classes_label):
        batch_new[batch == class_label] = dict_new_label[get_group(classes[i])]
    assert (batch_new.unique().int() > 9).sum() == 0
    return batch_new#.tolist() # convert tensor back to a list


#convert image with classes_label in dict_new_label
def convert_image(image):
    image_new = torch.zeros(image.shape)
    for i in range(len(classes_label)):
        image_new[image == classes_label[i]] = dict_new_label[get_group(classes[i])]
    assert (image_new.unique().int() > 9).sum() == 0
    return image_new

def get_group(class_name):
    for key in dict_group_classes:
        if class_name in dict_group_classes[key]:
            return key