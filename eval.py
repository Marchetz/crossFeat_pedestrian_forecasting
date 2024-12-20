
import argparse
import numpy as np
import torch
import random
import json
random.seed(99)
np.random.seed(99)
torch.manual_seed(99)
torch.cuda.manual_seed_all(99)
torch.set_printoptions(sci_mode=False)
from sklearn.metrics import average_precision_score, classification_report
from src.dataset.trans.data import *
from src.dataset.loader import *
from src.model.model_crossFeat import *
from src.transform.preprocess_seg import *
import matplotlib.colors as colors
from src.utils import *
from tqdm import tqdm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sklearn.metrics


def get_args():
    parser = argparse.ArgumentParser(description='Train hybrid model')
    parser.add_argument('--jaad', default=False, action='store_true',
                        help='use JAAD dataset')
    parser.add_argument('--pie', default=False, action='store_true',
                        help='use PIE dataset')
    parser.add_argument('--titan', default=False, action='store_true',
                        help='use TITAN dataset')
    parser.add_argument('--mode', default='GO', type=str,
                        help='transition mode, GO or STOP')
    parser.add_argument('--fps', default=5, type=int,
                        metavar='FPS', help='sampling rate(fps)')
    parser.add_argument('--max-frames', default=5, type=int,
                        help='maximum number of frames in histroy sequence')
    parser.add_argument('--pred', default=10, type=int,
                        help='prediction length, predicting-ahead time')
    parser.add_argument('--balancing_ratio', default=None, type=float,
                        help='ratio of balanced instances(1/0)')
    parser.add_argument('--seed', default=99, type=int,
                        help='random seed for sampling')
    parser.add_argument('--jitter-ratio', default=2.0, type=float,
                        help='jitter bbox for cropping')
    parser.add_argument('--bbox-min', default=24, type=int,
                        help='minimum bbox size')

    parser.add_argument('--sampler_flag', default=False, type=bool, help='-')
    parser.add_argument('--jb_flag', default=True, type=bool, help='-')

    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-e', '--epochs', default=200, type=int,
                        help='number of epochs to train')
    parser.add_argument('-wd', '--weight-decay', metavar='WD', type=float, default=1e-3,
                        help='Weight decay', dest='wd')
    parser.add_argument('-o', '--output', default='SEM_jaad_STOP_new',
                        help='output file')

    parser.add_argument('--model_pretrained_path', default='output_official/DECODER_jaad_go_mlp_H4_H1/epoch_best_test.pt', type=str,
                        help='path to encoder checkpoint for loading the pretrained weights')
    parser.add_argument('-n', '--n_trials', type=int, default=10, help='set number of trials for testing')

    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--feedforward_dim', type=int, default=2048)
    parser.add_argument('--sine_flag', type=bool, default=False)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_encoder_layers', type=int, default=1)
    parser.add_argument('--num_decoder_layers', type=int, default=1)
    parser.add_argument('--dropout_cnn', type=float, default=0.5)
    parser.add_argument('--dropout_transformer', type=float, default=0.5)
    parser.add_argument('--rgb', action='store_true', default=False)
    parser.add_argument('--freeze_resnet', action='store_true', default=False)
    parser.add_argument('--pretrained_rgb', default=False)
    parser.add_argument('--th_bbox', default=0.5, type=float)
    parser.add_argument('--type_seg', default='semantic', choices=['semantic', 'panoptic'])
    parser.add_argument('--bbox_generated', default='classic', choices=['classic', 'present', 'present_past'])


    args = parser.parse_args()


    return args

def draw(image, bb):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Create a Rectangle patch
    rect = patches.Rectangle((bb[0], bb[1]), (bb[2] - bb[0]), (bb[3] - bb[1]), linewidth=2, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


def val_epoch(phase, loader, model, device, args, count=0):
    encoder_CNN = model['encoder']
    encoder_CNN.eval()
    id_total = []
    y_true = []
    y_pred = []
    y_bbox = []
    y_bbox_true = []
    tte_total = []
    len_future_qual = []
    image_path_total = []


    with torch.no_grad():

        for i, inputs in tqdm(enumerate(loader)):

            id_total.extend(inputs['id'])
            tte_total.extend(inputs['tte'])
            len_future_qual.extend(inputs['len_future_qualitative'])

            if args.rgb:
                image_input = inputs['img_or'].permute(0, 3, 1, 2).float().to(device, non_blocking=True)
            else:
                images = inputs['image']
                one_hot = torch.zeros(images.shape[0], 10, images.shape[1], images.shape[2])
                one_hot = one_hot.scatter_(1, images.long().unsqueeze(1), 1)
                image_input = one_hot.to(device, non_blocking=True)

            bs = inputs['image'].shape[0]
            targets = inputs['label'].to(device, non_blocking=True)

            bboxes_history = inputs['bbox_history_norm']
            behavior_list = reshape_anns(inputs['behavior'], device)
            behavior = batch_first(behavior_list).to(device, non_blocking=True)
            scene = inputs['attributes'].to(device, non_blocking=True)
            bbox_ped_list = reshape_bbox_original(bboxes_history, device)
            pv = bbox_to_pv(bbox_ped_list).to(device, non_blocking=True)
            #
            seq_length = inputs['seq_length']
            bboxes_ped = inputs['bbox_ped']
            bbox_ped_list = reshape_bbox(bboxes_ped, device)
            bbox_last = torch.cat((torch.arange(bs).unsqueeze(1).unsqueeze(1).unsqueeze(1), bbox_ped_list), 3).squeeze(
                1).to(device, non_blocking=True)
            bbox_last = bbox_last.squeeze(1)
            speed = torch.stack(inputs['speed']).permute(1, 0).to(device, non_blocking=True)
            pose = None

            outputs_CNN = encoder_CNN(image_input, bbox_last, pv, behavior, scene, pose, speed, seq_length,
                                      training_flag=False)
            for j in range(targets.shape[0]):
                y_true.append(int(targets[j].item()))
                y_pred.append(float(outputs_CNN[j].item()))



    label_pred = (torch.Tensor(y_pred) > 0.5).int()
    precision = sklearn.metrics.precision_score(y_true, label_pred)
    accuracy = sklearn.metrics.accuracy_score(y_true, label_pred)
    recall = sklearn.metrics.recall_score(y_true, label_pred)
    F1 = sklearn.metrics.f1_score(y_true, label_pred)
    AP_P = average_precision_score(y_true, y_pred)

    y_pred = torch.Tensor(y_pred)
    y_true = torch.Tensor(y_true)
    tte_total = torch.stack(tte_total)
    len_future_qual = torch.stack(len_future_qual)


    # print('time_inference: ', np.mean(time_inference))
    metrics = {'AP': AP_P,
               'accuracy': accuracy,
               'precision': precision,
               'recall': recall,
               'f1': F1}
    data = {'ids': id_total,
            'y_true': y_true,
            'y_pred': label_pred,
            'y_pred_values': y_pred,
            'tte': tte_total,
            'len_future_qual': len_future_qual,
            'y_bbox': y_bbox,
            'y_bbox_true': y_bbox_true}

    return AP_P, metrics, data

def inference(data, model, device):
    with torch.no_grad():
        inputs = data
        images = inputs['image'].unsqueeze(0)
        speed = torch.Tensor(inputs['speed']).unsqueeze(0).to(device, non_blocking=True)
        one_hot = torch.zeros(images.shape[0], 10, images.shape[1], images.shape[2])
        one_hot = one_hot.scatter_(1, images.long().unsqueeze(1), 1)
        one_hot_image = one_hot.to(device, non_blocking=True)

        bs = images.shape[0]
        targets = inputs['label'].unsqueeze(0).to(device, non_blocking=True)

        #inputs['bbox_history_norm']
        temp = []
        for i in range(len(inputs['bbox_history_norm'])):
            temp_j = []
            for j in range(len(inputs['bbox_history_norm'][i])):
                temp_j.append(torch.Tensor([inputs['bbox_history_norm'][i][j]]))
            temp.append(temp_j)
        bboxes_history = temp

        #same with behavior
        temp = []
        for i in range(len(inputs['behavior'])):
            temp_j = []
            for j in range(len(inputs['behavior'][i])):
                temp_j.append(torch.Tensor([inputs['behavior'][i][j]]))
            temp.append(temp_j)
        behavior_list = reshape_anns(temp, device)
        behavior = batch_first(behavior_list).to(device, non_blocking=True)

        scene = inputs['attributes'].unsqueeze(0).to(device, non_blocking=True)
        bbox_ped_list = reshape_bbox_original(bboxes_history, device)
        pv = bbox_to_pv(bbox_ped_list).to(device, non_blocking=True)
        seq_length = [inputs['seq_length']]
        seq_length = torch.Tensor(seq_length)
        seq_length = seq_length.long()

        bboxes_ped = torch.Tensor(inputs['bbox_ped']).unsqueeze(0)
        temp = []
        for i in range(len(inputs['bbox_ped'])):
            temp_j = []
            for j in range(len(inputs['bbox_ped'][i])):
                temp_j.append(torch.Tensor([inputs['bbox_ped'][i][j]]))
            temp.append(temp_j)
        bbox_ped_list = reshape_bbox(temp, device)
        bbox_last = torch.cat((torch.arange(bs).unsqueeze(1).unsqueeze(1).unsqueeze(1), bbox_ped_list), 3).squeeze(
            1).to(device, non_blocking=True)
        bbox_last = bbox_last.squeeze(1)
        pose = None
        outputs = model(one_hot_image, bbox_last, pv, behavior, scene, pose, speed, seq_length)
        return outputs


def main():
    if sys.gettrace() is not None:
        num_workers = 0
        debug_frame = True
    else:
        num_workers = 8
        debug_frame = False
    args = get_args()
    model_pretrained_path = args.model_pretrained_path
    n_trials = args.n_trials

    th_bbox = args.th_bbox
    type_seg = args.type_seg
    bbox_generated = args.bbox_generated

    jaad_flag = args.jaad
    pie_flag = args.pie
    titan_flag = args.titan
    batch_size = args.batch_size

    with open(model_pretrained_path + '/args.json', 'r') as f:
        args_temp = json.load(f)
    args = argparse.Namespace(**args_temp)
    args.jaad = jaad_flag
    args.pie = pie_flag
    args.titan = titan_flag
    args.batch_size = batch_size
    args.type_seg = type_seg
    args.th_bbox = th_bbox
    args.bbox_generated = bbox_generated

    #set model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_crossFeat(args)
    model = model.to(device)
    #load checkpoint
    checkpoint = torch.load(f'{model_pretrained_path}/model_pretrained.pt')
    model.load_state_dict(checkpoint['model'])
    model_gpu = {'encoder': model}
    print('AP: ' + str(checkpoint['AP']))

    # loading data
    print('Start annotation loading -->', 'JAAD:', args.jaad, 'PIE:', args.pie, 'TITAN:', args.titan)
    anns_paths, image_dir = define_path(use_jaad=args.jaad, use_pie=args.pie, use_titan=args.titan)
    print('------------------------------------------------------------------')

    train_data = TransDataset(data_paths=anns_paths, image_set="train", verbose=False)
    trans_tr = train_data.extract_trans_history(mode=args.mode, fps=args.fps, max_frames=None, verbose=False)
    non_trans_tr = train_data.extract_non_trans(fps=5, max_frames=None, verbose=True)
    print('-->>')
    val_data = TransDataset(data_paths=anns_paths, image_set="val", verbose=False)
    trans_val = val_data.extract_trans_history(mode=args.mode, fps=args.fps, max_frames=None, verbose=False)
    non_trans_val = val_data.extract_non_trans(fps=5, max_frames=None, verbose=True)
    print('-->>')
    test_data = TransDataset(data_paths=anns_paths, image_set="test", verbose=False)
    trans_test = test_data.extract_trans_history(mode=args.mode, fps=args.fps, max_frames=None, verbose=False)
    non_trans_test = test_data.extract_non_trans(fps=5, max_frames=None, verbose=True)
    print('-->>')

    crop_preprocess = DownSizeFrame(1281, 481)

    if args.jb_flag:
        TRANSFORM = Compose([crop_preprocess, JitterBox(args.jitter_ratio)])
        TRANSFORM_1 = Compose([crop_preprocess])
        TRANSFORM_2 = Compose([JitterBox(args.jitter_ratio)])
    else:
        TRANSFORM = crop_preprocess
    loader_val_total = []
    loader_test_total = []
    sequences_test_all = []

    for n_t in range(n_trials):
        print('-->> trial: ' + str(n_t))

        print('train eval')
        sequences_train_eval = extract_pred_sequence(trans=trans_tr, non_trans=non_trans_tr, pred_ahead=args.pred,
                                                     balancing_ratio=1.0, neg_in_trans=True,
                                                     bbox_min=args.bbox_min, max_frames=args.max_frames, seed=args.seed,
                                                     verbose=True)
        print('train')
        sequences_train = extract_pred_sequence(trans=trans_tr, non_trans=non_trans_tr, pred_ahead=args.pred,
                                                balancing_ratio=1.0, neg_in_trans=True,
                                                bbox_min=args.bbox_min, max_frames=args.max_frames, seed=args.seed,
                                                verbose=True)
        print('val')
        sequences_val = extract_pred_sequence(trans=trans_val, non_trans=non_trans_val, pred_ahead=args.pred,
                                              balancing_ratio=1.0, neg_in_trans=True,
                                              bbox_min=args.bbox_min, max_frames=args.max_frames, seed=args.seed,
                                              verbose=True)
        print('test')
        sequences_test = extract_pred_sequence(trans=trans_test, non_trans=non_trans_test, pred_ahead=args.pred,
                                               balancing_ratio=1.0, neg_in_trans=True,
                                               bbox_min=args.bbox_min, max_frames=args.max_frames, seed=args.seed,
                                               verbose=True)
        sequences_test_all.append(sequences_test)


        val_instances = PaddedSequenceDataset_segSem('val', sequences_val, image_dir=image_dir, padded_length=args.max_frames,
                                                     hflip_p=0.0, preprocess=TRANSFORM, depth=None, debug=debug_frame, args=args, preprocess_1= TRANSFORM_1, preprocess_2= TRANSFORM_2)
        val_loader = torch.utils.data.DataLoader(val_instances, num_workers=num_workers, batch_size=args.batch_size, shuffle=False,
                                                 pin_memory=True)
        loader_val_total.append(val_loader)

        test_instances = PaddedSequenceDataset_segSem('test', sequences_test, image_dir=image_dir,
                                                      padded_length=args.max_frames, hflip_p=0.0, preprocess=TRANSFORM,
                                                      depth=None, debug=debug_frame, args=args, preprocess_1= TRANSFORM_1, preprocess_2= TRANSFORM_2)
        test_loader = torch.utils.data.DataLoader(test_instances, num_workers=num_workers, batch_size=args.batch_size, shuffle=False,
                                                  pin_memory=True)
        loader_test_total.append(test_loader)
        print('------------------------------------------------------------')
    print('------------------------------------------------------------------')
    print('Finish annotation loading', '\n')


    print('quantitative phase')
    count = 0
    AP_val_total = []
    AP_test_total = []
    accuracy_test_total = []
    precision_test_total = []
    recall_test_total = []
    f1_test_total = []
    decoder_attention_total = []
    for n_t in range(n_trials):
        #AP_val, metrics_val, data = val_epoch('val', loader_val_total[n_t], model_gpu, device)
        AP_test, metrics_test, data = val_epoch('test', loader_test_total[n_t], model_gpu, device, args, count)
        #AP_val_total.append(AP_val)

        AP_test_total.append(AP_test)
        accuracy_test_total.append(round(metrics_test['accuracy'], 2))
        precision_test_total.append(round(metrics_test['precision'], 2))
        recall_test_total.append(round(metrics_test['recall'], 2))
        f1_test_total.append(round(metrics_test['f1'], 2))
        count += 1

    print('path: ', model_pretrained_path)
    print('mode -->', args.mode)
    print('dataset used -->', 'JAAD:', args.jaad, 'PIE:', args.pie, 'TITAN:', args.titan)
    #print(AP_val_total)
    print('AP test -->', AP_test_total)
    print('accuracy test -->', accuracy_test_total)
    print('precision test -->', precision_test_total)
    print('recall test -->', recall_test_total)
    print('f1 test -->', f1_test_total)



    print(f'average AP test {round(np.mean(AP_test_total),3)}')
    print(f'average accuracy test {round(np.mean(accuracy_test_total),3)}')
    print(f'average precision test {round(np.mean(precision_test_total),3)}')
    print(f'average recall test {round(np.mean(recall_test_total),3)}')
    print(f'average f1 test {round(np.mean(f1_test_total),3)}')
    print('\n')


    print('--------------------------------------------------------', '\n')
    print('\n', '**************************************************************')


if __name__ == '__main__':
    main()
