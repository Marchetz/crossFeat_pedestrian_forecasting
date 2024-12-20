import argparse
import time
import comet_ml
import torch
import sklearn.metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import numpy as np
import random
import json
random.seed(99)
np.random.seed(99)
torch.manual_seed(99)
torch.cuda.manual_seed_all(99)
from tqdm import tqdm
from PIL import Image, ImageFile
from src.dataset.loader import *
from src.model.model_crossFeat import *
from src.transform.preprocess_seg import *
from src.utils import *
from src.losses_imb import *
from src.dataset.info_data import *
ImageFile.LOAD_TRUNCATED_IMAGES = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

g = torch.Generator()
g.manual_seed(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.cuda.amp import autocast, GradScaler
torch.set_printoptions(sci_mode=False)


#TODO: add own Comet parameters
API_KEY = "---"
PROJECT_NAME = "---"
WORKSPACE = "---"

def get_args():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--jaad', default=False, action='store_true', help='use JAAD dataset')
    parser.add_argument('--pie', default=False, action='store_true', help='use PIE dataset')
    parser.add_argument('--titan', default=False, action='store_true', help='use TITAN dataset')
    parser.add_argument('--loki', default=False, action='store_true', help='use LOKI dataset')
    parser.add_argument('--mode', default='GO', type=str, help='transition mode, GO or STOP')
    parser.add_argument('--fps', default=5, type=int, metavar='FPS', help='sampling rate(fps)')
    parser.add_argument('--max-frames', default=5, type=int, help='maximum number of frames in histroy sequence')
    parser.add_argument('--pred', default=10, type=int, help='prediction length, predicting-ahead time')
    parser.add_argument('--balancing-ratio', default=1.0, type=float, help='ratio of balanced instances(1/0)')
    parser.add_argument('--seed', default=99, type=int, help='random seed for sampling')
    parser.add_argument('--jitter-ratio', default=2.0, type=float, help='jitter bbox for cropping')
    parser.add_argument('--bbox_min', default=24, type=int, help='minimum bbox size')

    parser.add_argument('--sampler_flag', default=False, type=bool, help='-')
    parser.add_argument('--jb_flag', default=True, type=bool, help='-')
    parser.add_argument('--model_path', default=None, type=str, help='-')

    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-b_val', '--batch_size_val', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-e', '--epochs', default=100, type=int,
                        help='number of epochs to train')
    parser.add_argument('-wd', '--weight-decay', metavar='WD', type=float, default=1e-5,
                        help='Weight decay', dest='wd')
    parser.add_argument('-o', '--output', default='sem_prova',
                        help='output file')

    parser.add_argument('--hidden_dim', type=int, default=64) #64
    parser.add_argument('--feedforward_dim', type=int, default=256)
    parser.add_argument('--sine_flag', type=bool, default=False)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_encoder_layers', type=int, default=1)
    parser.add_argument('--num_decoder_layers', type=int, default=1)
    parser.add_argument('--dropout_cnn', type=float, default=0.5)
    parser.add_argument('--dropout_transformer', type=float, default=0.1)
    parser.add_argument('--weighted_loss', type=bool, default=True)
    parser.add_argument('--all_val_test', type=bool, default=False)
    parser.add_argument('--type_loss', type=str, default='binary_cross_entropy', choices=['binary_cross_entropy', 'focal_loss'])
    parser.add_argument('--paper_version', type=bool, default=False)

    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--freeze_resnet', action='store_true')
    parser.add_argument('--pretrained_rgb',  action='store_true')
    parser.add_argument('--th', type=float, default=0.5)
    parser.add_argument('--after_roi', action='store_true', default=False)
    parser.add_argument('--mask', action='store_true', default=False)

    parser.add_argument('--load_args', default=None)
    parser.add_argument('--type_seg', default='classic', choices=['classic', 'CS_R50', 'DLv3'])
    parser.add_argument('--fp16_flag', action='store_true', default=False)

    parser.add_argument('--dataset', type=str, default='pie', choices=['pie', 'jaad_all', 'jaad_beh'])  #configs/configs_all.yaml, configs_beh.yaml

    args = parser.parse_args()

    return args

def draw(image, bbox):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    id = 6
    ax.imshow(image.cpu().permute(1, 2, 0))

    # Create a Rectangle patch
    bb = bbox[1:].cpu()
    rect = patches.Rectangle((bb[0], bb[1]), (bb[2] - bb[0]), (bb[3] - bb[1]), linewidth=1, edgecolor='r',
                             facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

def analysis(data, counts, num_examples):

    images = data['image']
    num_labels = 66
    for i in range(num_labels):
        counts[i] += (images == i).sum()
    num_examples += images.shape[0]
    return counts, num_examples

def count_parameters(model):
    num_params = 0
    for p in model.parameters():
        if p.requires_grad:
            #print(p.numel())
            num_params += p.numel()
    print("Number of parameters:", num_params)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def prepare_data(inputs, args, device, processor):

    size_channels = 10
    if args.rgb:
        image_input = inputs['img_or'].permute(0, 3, 1, 2).float().to(device, non_blocking=True)
    else:
        images = inputs['image']
        one_hot = torch.zeros(images.shape[0], size_channels, images.shape[1], images.shape[2])
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
    bbox_last = torch.cat((torch.arange(bs).unsqueeze(1).unsqueeze(1).unsqueeze(1), bbox_ped_list), 3).squeeze(1).to(
        device, non_blocking=True)
    bbox_last = bbox_last.squeeze(1)

    speed = torch.stack(inputs['speed']).permute(1, 0).to(device, non_blocking=True)
    pose = None

    return image_input, bbox_last, pv, behavior, scene, pose, speed, seq_length, targets

def train_epoch(loader, model, criterion, optimizer, device, class_data, args, processor):
    encoder_CNN = model['encoder']
    encoder_CNN.train()
    scaler = GradScaler()
    epoch_loss = 0.0
    total_examples = 0
    total_batch = 0


    for i, inputs in tqdm.tqdm(enumerate(loader)):

        optimizer.zero_grad(set_to_none=True)
        image_input, bbox_last, pv, behavior, scene, pose, speed, seq_length, targets = prepare_data(inputs, args, device, processor)
        bs = pv.shape[0]
        # with autocast():
        outputs_CNN = encoder_CNN(image_input, bbox_last, pv, behavior, scene, pose, speed, seq_length)
        loss = criterion(outputs_CNN, targets)

        loss_value = float(loss.item())


        epoch_loss += loss_value
        total_examples += bs
        total_batch += 1

        reg_strength = 0.001
        l2_reg = torch.tensor(0.).cuda()
        for param in encoder_CNN.fc_last.parameters():
            l2_reg += torch.norm(param, 2)
        loss += reg_strength * l2_reg

        loss.backward()
        optimizer.step()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

    loss_train = epoch_loss / total_batch

    return loss_train


def val_epoch(phase, loader, model, criterion, device, epoch, args, processor):
    encoder_CNN = model['encoder']
    encoder_CNN.eval()
    epoch_loss = 0.0
    y_true = []
    y_pred = []
    total_examples = 0
    total_batch = 0

    with torch.no_grad():
        for i, inputs in tqdm.tqdm(enumerate(loader)):
            image_input, bbox_last, pv, behavior, scene, pose, speed, seq_length, targets = prepare_data(inputs, args, device, processor)
            bs = pv.shape[0]
            outputs_CNN = encoder_CNN(image_input, bbox_last, pv, behavior, scene, pose, speed, seq_length)

            loss = criterion(outputs_CNN, targets)
            epoch_loss += float(loss.item())

            total_examples += bs
            total_batch += 1
            for j in range(targets.shape[0]):
                y_true.append(int(targets[j].item()))
                y_pred.append(float(outputs_CNN[j].item()))

    label_pred = (torch.Tensor(y_pred) > 0.5).int()
    precision = sklearn.metrics.precision_score(y_true, label_pred)
    accuracy = sklearn.metrics.accuracy_score(y_true, label_pred)
    recall = sklearn.metrics.recall_score(y_true, label_pred)
    F1 = sklearn.metrics.f1_score(y_true, label_pred)
    AP_P = average_precision_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    loss = epoch_loss / total_batch

    metrics = {'AP': AP_P,
               'accuracy': accuracy,
               'precision': precision,
               'recall': recall,
               'f1': F1,
               'auc_roc': auc_roc,
               'loss': loss,
               }
    data = {'y_true': y_true,
            'y_pred': label_pred}

    return loss, AP_P, metrics, data  #, image_path_total#, counts, num_examples



def main():
    args = get_args()
    output_str = args.output
    type_seg = args.type_seg
    fp16_flag = args.fp16_flag
    hidden_dim = args.hidden_dim

    print('mode: ' + args.mode)

    if args.load_args != None:
        with open(args.load_args, 'r') as f:
            args_temp = json.load(f)
        args = argparse.Namespace(**args_temp)
        args.type_seg = type_seg
        args.fp16_flag = fp16_flag
        args.hidden_dim = hidden_dim
        args.pretrained_rgb = False

    args.output = output_str

    if not os.path.exists('output/' + args.output):
        os.makedirs('output/' + args.output)
    Save_path = 'output/' + args.output

    with open(Save_path + '/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if sys.gettrace() is not None:
        n_work = 0
        disable_comet = True
        debug_frame = True
    else:
        n_work = 4
        disable_comet = False
        debug_frame = False

    log_writer = comet_ml.Experiment(api_key=API_KEY, project_name=PROJECT_NAME,
                                          workspace=WORKSPACE, display_summary_level=0,
                                          disabled=disable_comet,
                                          log_code=True, auto_metric_logging=True)
    log_writer.set_name(args.output)
    log_writer.log_parameters(args)
    log_writer.log_code(file_name='src/model/model_crossFeat.py')
    log_writer.log_code(file_name='src/dataset/loader.py')
    log_writer.log_code(file_name='src/losses_imb.py')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_crossFeat(args)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path)['model'])
    model = model.to(device)
    count_parameters(model)
    model_gpu = {'encoder': model}

    # training settings
    crnn_params = list(model.parameters())
    optimizer = torch.optim.AdamW(crnn_params, lr=args.lr, weight_decay=args.wd)
    if args.model_path is not None:
        optimizer.load_state_dict(torch.load(args.model_path)['optimizer'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    start_epoch = 0
    end_epoch = start_epoch + args.epochs

    # loading data
    print('Start annotation loading -->', 'JAAD:', args.jaad, 'PIE:', args.pie, 'TITAN:', args.titan, 'LOKI:', args.loki)
    anns_paths, image_dir = define_path(use_jaad=args.jaad, use_pie=args.pie, use_titan=args.titan)
    print('------------------------------------------------------------------')

    train_data = TransDataset(data_paths=anns_paths, image_set="train", verbose=True)
    trans_tr = train_data.extract_trans_history(mode=args.mode, fps=args.fps, max_frames=None, verbose=True)
    non_trans_tr = train_data.extract_non_trans(fps=5, max_frames=None, verbose=True)
    print('-->>')
    #in the original code for stop&go benchmark, the val phase is made with test dataset, so we calculate the metrics for both sets and save the best models.
    val_data = TransDataset(data_paths=anns_paths, image_set="val", verbose=True)
    trans_val = val_data.extract_trans_history(mode=args.mode, fps=args.fps, max_frames=None, verbose=True)
    non_trans_val = val_data.extract_non_trans(fps=5, max_frames=None, verbose=True)
    print('-->>')
    test_data = TransDataset(data_paths=anns_paths, image_set="test", verbose=True)
    trans_test = test_data.extract_trans_history(mode=args.mode, fps=args.fps, max_frames=None, verbose=True)
    non_trans_test = test_data.extract_non_trans(fps=5, max_frames=None, verbose=True)
    print('-->>')

    if args.weighted_loss:
        balancing_ratio = None
    else:
        balancing_ratio = args.balancing_ratio

    if args.all_val_test:
        balancing_ratio_val = None
    else:
        balancing_ratio_val = args.balancing_ratio

    #extract_pred_sequence
    sequences_train_eval = extract_pred_sequence(trans=trans_tr, non_trans=non_trans_tr, pred_ahead=args.pred,
                                            balancing_ratio=1.0, neg_in_trans=True,
                                            bbox_min=args.bbox_min, max_frames=args.max_frames, seed=args.seed,
                                            verbose=True)
    log_writer.log_text(list(sequences_train_eval.keys()))
    sequences_train = extract_pred_sequence(trans=trans_tr, non_trans=non_trans_tr, pred_ahead=args.pred,
                                            balancing_ratio=balancing_ratio, neg_in_trans=True,
                                            bbox_min=args.bbox_min, max_frames=args.max_frames, seed=args.seed, verbose=True)
    print('-->>')
    sequences_val = extract_pred_sequence(trans=trans_val, non_trans=non_trans_val, pred_ahead=args.pred,
                                          balancing_ratio=balancing_ratio_val, neg_in_trans=True,
                                          bbox_min=args.bbox_min, max_frames=args.max_frames, seed=args.seed, verbose=True)
    log_writer.log_text(list(sequences_val.keys()))
    print('------------------------------------------------------------------')
    sequences_test = extract_pred_sequence(trans=trans_test, non_trans=non_trans_test, pred_ahead=args.pred,
                                          balancing_ratio=balancing_ratio_val, neg_in_trans=True,
                                          bbox_min=args.bbox_min, max_frames=args.max_frames, seed=args.seed, verbose=True)
    log_writer.log_text(list(sequences_test.keys()))

    print('------------------------------------------------------------------')
    print('Finish annotation loading', '\n')

    if args.weighted_loss:
        class_data = {}
        if args.jaad and args.pie and args.titan:
            pos_total = 0
            neg_total = 0
            total_samples = 0
            for d in ['JAAD', 'PIE', 'TITAN']:
                pos = examples_count[d][args.mode]['pos']
                neg = examples_count[d][args.mode]['neg']
                pos_total += pos
                neg_total += neg
                total_samples += neg + pos
            class_freq = [neg_total / total_samples, pos_total / total_samples]
            median_freq = torch.median(torch.tensor(class_freq))

            # Calculate the class weights
            class_weights = torch.tensor([median_freq / freq for freq in class_freq])
            class_number = [neg, pos]
            class_data['class_weights'] = class_weights
            class_data['class_number'] = class_number
        else:
            if args.jaad: dataset_choice = 'JAAD'
            elif args.pie: dataset_choice = 'PIE'
            elif args.titan: dataset_choice = 'TITAN'
            pos = examples_count[dataset_choice][args.mode]['pos']
            neg = examples_count[dataset_choice][args.mode]['neg']
            total_samples = neg + pos
            class_freq = [neg / total_samples, pos / total_samples]
            median_freq = torch.median(torch.tensor(class_freq))

            # Calculate the class weights
            class_weights = torch.tensor([median_freq / freq for freq in class_freq])
            class_number = [neg, pos]

            class_freq = torch.Tensor([pos / total_samples, neg / total_samples])
            class_data['class_weights'] = class_freq
            class_data['class_number'] = class_number

    else:
        class_data = {}
        class_data['class_number'] = None
        class_data['class_weights'] = None
    criterion = Loss_imb(
        loss_type=args.type_loss,   #binary_cross_entropy, focal_loss
        samples_per_class=class_data['class_number'],
        samples_weight=class_data['class_weights'],
        class_balanced=True,
        paper_version=args.paper_version
    )
    # criterion = nn.BCEWithLogitsLoss()

    crop_preprocess = DownSizeFrame(1281, 481)
    if args.jb_flag:
        # TRANSFORM_TRAIN = Compose([crop_preprocess,  ImageTransform(torchvision.transforms.ColorJitter(
        #                            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)), JitterBox(args.jitter_ratio)])
        TRANSFORM_TRAIN = Compose([crop_preprocess, JitterBox(args.jitter_ratio)])
        TRANSFORM_VAL = Compose([crop_preprocess, JitterBox(args.jitter_ratio)])
    else:
        TRANSFORM_TRAIN = crop_preprocess
        TRANSFORM_VAL = crop_preprocess

    train_instances = PaddedSequenceDataset_segSem('train', sequences_train, image_dir=image_dir, padded_length=args.max_frames, hflip_p=0.5, preprocess=TRANSFORM_TRAIN, depth=None, debug=debug_frame, args=args)
    train_instances_eval = PaddedSequenceDataset_segSem('eval_train', sequences_train_eval, image_dir=image_dir, padded_length=args.max_frames, hflip_p=0.0, preprocess=TRANSFORM_VAL, depth=None, debug=debug_frame, args=args)
    val_instances = PaddedSequenceDataset_segSem('val', sequences_val, image_dir=image_dir, padded_length=args.max_frames, hflip_p=0.0, preprocess=TRANSFORM_VAL, depth=None, debug=debug_frame, args=args)
    test_instances = PaddedSequenceDataset_segSem('test', sequences_test, image_dir=image_dir, padded_length=args.max_frames, hflip_p=0.0, preprocess=TRANSFORM_VAL, depth=None, debug=debug_frame, args=args)

    path_total = []
    for path in train_instances.path_image:
        path_total.append(path)
    for path in val_instances.path_image:
        path_total.append(path)
    for path in test_instances.path_image:
        path_total.append(path)
    for path in train_instances_eval.path_image:
        path_total.append(path)
    #remove duplicate
    path_total = list(set(path_total))
    #save path_total as pickle
    with open('path_total.pkl', 'wb') as f:
        pickle.dump(path_total, f)

    sampler_train = None
    shuffle_train = True

    pin_memory_flag = True
    train_loader = torch.utils.data.DataLoader(train_instances, num_workers=n_work, batch_size=args.batch_size, shuffle=shuffle_train, pin_memory=pin_memory_flag, sampler=sampler_train, worker_init_fn=seed_worker, generator=g, drop_last=True)
    train_loader_eval = torch.utils.data.DataLoader(train_instances_eval, num_workers=n_work, batch_size=args.batch_size_val, shuffle=False, pin_memory=pin_memory_flag, sampler=None, worker_init_fn=seed_worker, generator=g)
    val_loader = torch.utils.data.DataLoader(val_instances, num_workers=n_work, batch_size=args.batch_size_val, shuffle=False, pin_memory=pin_memory_flag, sampler=None, worker_init_fn=seed_worker, generator=g)
    test_loader = torch.utils.data.DataLoader(test_instances, num_workers=n_work, batch_size=args.batch_size_val, shuffle=False, pin_memory=pin_memory_flag, sampler=None, worker_init_fn=seed_worker, generator=g)

    log_writer.log_other('len_TRAIN_dataset', len(train_instances))
    log_writer.log_other('len_TRAIN_EVAL_dataset', len(train_instances_eval))
    log_writer.log_other('len_VAL_dataset', len(val_instances))
    log_writer.log_other('len_TEST_dataset', len(test_instances))
    print(f'train examples: {len(train_instances)}')
    print(f'train eval examples : {len(train_instances_eval)}')
    print(f'val examples : {len(val_instances)}')
    print(f'test examples : {len(test_instances)}')

    print(f'train loader : {len(train_loader)}')
    print(f'train eval loader : {len(train_loader_eval)}')
    print(f'val loader : {len(val_loader)}')
    print(f'test loader : {len(test_loader)}')
    total_time = 0.0
    ap_min_val = 0.0
    ap_min_test = 0.0
    print(
        f'Start training, PVIBS-lstm-model, neg_in_trans, initail lr={args.lr}, weight-decay={args.wd}, mf={args.max_frames}, training batch size={args.batch_size}')


    # #SEGMENTATION MODULE
    # processor = AutoImageProcessor.from_pretrained(pretrained_model)
    processor = None


    for epoch in range(start_epoch, end_epoch):
        print(f'train loader : {len(train_loader)}')
        log_writer.log_other('len_TRAIN_dataset', len(train_instances))

        print(f'Start of epoch {epoch}, {args.output}')
        start_epoch_time = time.time()
        train_loss = train_epoch(train_loader, model_gpu, criterion, optimizer, device, class_data, args, processor)
        log_writer.log_metric('loss', train_loss, epoch=epoch)

        # with log_writer.train():
        #     train_loss, val_score, metrics_train, data = val_epoch('train', train_loader_eval, model_gpu, criterion, device, epoch, args, processor)
        #     log_writer.log_metrics(metrics_train, epoch=epoch)
        # log_writer.log_confusion_matrix(data['y_true'], data['y_pred'], file_name='confusion_train', epoch=epoch)
        with log_writer.validate():
            val_loss, val_score, metrics_val, data = val_epoch('val', val_loader, model_gpu, criterion, device, epoch, args, processor)
            log_writer.log_metrics(metrics_val, epoch=epoch)
        log_writer.log_confusion_matrix(data['y_true'], data['y_pred'], file_name='confusion_val', epoch=epoch)
        with log_writer.test():
            test_loss, test_score, metrics_test, data = val_epoch('test', test_loader, model_gpu, criterion, device, epoch, args, processor)
            log_writer.log_metrics(metrics_test, epoch=epoch)
        log_writer.log_confusion_matrix(data['y_true'], data['y_pred'], file_name='confusion_test', epoch=epoch)

        scheduler.step(test_score)
        #scheduler = 0
        end_epoch_time = time.time() - start_epoch_time
        print('\n')

        print('Validation epoch AP score: {:.4f}'.format(val_score))
        print('Test epoch AP score: {:.4f}'.format(test_score))
        print('Test epoch ACC score: {:.4f}'.format(metrics_test['accuracy']))
        print('Test epoch F1 score: {:.4f}'.format(metrics_test['f1']))
        print('Test epoch AUC_ROC score: {:.4f}'.format(metrics_test['auc_roc']))
        print(f'End of epoch {epoch}')
        print('--------------------------------------------------------', '\n')
        total_time += end_epoch_time
        if val_score > ap_min_val:
            # save_to_checkpoint(Save_path, epoch, model_gpu['encoder'], optimizer, scheduler, verbose=True, info='val', AP=val_score)
            save_to_checkpoint(Save_path, 'best', model_gpu['encoder'], optimizer, scheduler, verbose=True, info='val',
                               AP=val_score)
            ap_min_val = val_score
        if test_score > ap_min_test:
            # save_to_checkpoint(Save_path, epoch, model_gpu['encoder'], optimizer, scheduler, verbose=True, info='test', AP=test_score)
            save_to_checkpoint(Save_path, 'best', model_gpu['encoder'], optimizer, scheduler, verbose=True, info='test',
                               AP=test_score)
            ap_min_test = test_score

    print('\n', '**************************************************************')
    save_to_checkpoint(Save_path, end_epoch, model_gpu['encoder'], optimizer, scheduler, verbose=True, info='latest',
                       AP=test_score)
    print(f'End training at epoch {end_epoch}')
    print('total time: {:.2f}'.format(total_time))


# def main():
if __name__ == '__main__':
    main()
