from typing import List
from .jaad_trans_tam import *
from .pie_trans import *
from .titan_trans import *
from .loki_trans import *

import random
random.seed(99)


class TransDataset:
    """
    Unified class for using data from JAAD, PIE and TITAN dataset.
    """

    def __init__(self, data_paths, image_set="all", subset='default', verbose=False):
        dataset = {}
        assert image_set in ['train', 'test', 'val', "all"], " Name should be train, test, val or all"
        for d in list(data_paths.keys()):
            assert d in ['JAAD', 'PIE', 'TITAN', 'LOKI'], " Available datasets are JAAD, PIE and TITAN"
            if d == "JAAD":
                dataset['JAAD'] = JaadTransDataset(
                    jaad_anns_path=data_paths['JAAD']['anns'],
                    split_vids_path=data_paths['JAAD']['split'],
                    image_set=image_set,
                    subset=subset, verbose=verbose)
            elif d == "PIE":
                dataset['PIE'] = PieTransDataset(
                    pie_anns_path=data_paths['PIE']['anns'],
                    image_set=image_set, verbose=verbose)
            elif d == "TITAN":
                dataset['TITAN'] = TitanTransDataset(
                    anns_dir=data_paths['TITAN']['anns'],
                    split_vids_path=data_paths['TITAN']['split'],
                    image_set=image_set, verbose=verbose)
            elif d == "LOKI":
                dataset['LOKI'] = LokiTransDataset(
                    anns_dir=data_paths['LOKI']['anns'],
                    split_vids_path='',
                    image_set='all', verbose=True)

        self.dataset = dataset
        self.name = image_set
        self.subset = subset

    def __repr__(self):
        return f"TransDataset(image_set={self.name}, jaad_subset={self.subset})"

    def extract_trans_frame(self, mode="GO", frame_ahead=0, fps=10, verbose=False) -> dict:
        ds = list(self.dataset.keys())
        samples = {}
        for d in ds:
            samples_new = self.dataset[d].extract_trans_frame(mode=mode, frame_ahead=frame_ahead, fps=fps)
            samples.update(samples_new)
        if verbose:
            ids = list(samples.keys())
            pids = []
            for idx in ids:
                pids.append(samples[idx]['old_id'])
            print(f"Extract {len(pids)} {mode} frame samples from {self.name} dataset,")
            print(f"samples contain {len(set(pids))} unique pedestrians.")

        return samples

    def extract_trans_history(self, mode="GO", fps=10, max_frames=None, post_frames=0, verbose=False) -> dict:
        """
        Extract the whole history of pedestrian up to the frame when transition happens
        :params: mode: target transition type, "GO" or "STOP"
                fps: frame-per-second, sampling rate of extracted sequences, default 10
                max_frames: maximum number of frames in one history
                post_frames: number of frames included after the transition
                verbose: optional printing of sample statistics
        """
        assert isinstance(fps, int) and 30 % fps == 0, "impossible fps"
        ds = list(self.dataset.keys())
        samples = {}
        for d in ds:
            samples_new = self.dataset[d].extract_trans_history(mode=mode, fps=fps, max_frames=max_frames,
                                                                post_frames=post_frames)
            samples.update(samples_new)
        if verbose:
            ids = list(samples.keys())
            pids = []
            num_frames = 0
            for idx in ids:
                pids.append(samples[idx]['old_id'])
                num_frames += len(samples[idx]['frame'])
            print(f"Extract {len(pids)} {mode} history samples from {self.name} dataset,")
            print(f"samples contain {len(set(pids))} unique pedestrians and {num_frames} frames.")

        return samples

    def extract_non_trans(self, fps=10, max_frames=None, verbose=False) -> dict:
        assert isinstance(fps, int) and 30 % fps == 0, "impossible fps"
        ds = list(self.dataset.keys())
        samples = {'walking': {}, 'standing': {}}
        for d in ds:
            # Set the number of samples needed in TITAN
            if d == 'TITAN':
                if self.name == 'all':
                    n_titan = 600
                elif self.name == 'train':
                    n_titan = 300
                elif self.name == 'val':
                    n_titan = 200
                else:
                    n_titan = 100
                samples_new = self.dataset[d].extract_non_trans(fps=fps, max_frames=max_frames, max_samples=n_titan)
            else:
                samples_new = self.dataset[d].extract_non_trans(fps=fps, max_frames=max_frames)
            samples['walking'].update(samples_new['walking'])
            samples['standing'].update(samples_new['standing'])
        if verbose:
            keys_w = list(samples['walking'].keys())
            keys_s = list(samples['standing'].keys())
            pid_w = []
            pid_s = []
            n_w = 0
            n_s = 0
            for kw in keys_w:
                pid_w.append(samples['walking'][kw]['old_id'])
                n_w += len(samples['walking'][kw]['frame'])
            for ks in keys_s:
                pid_s.append(samples['standing'][ks]['old_id'])
                n_s += len(samples['standing'][ks]['frame'])
            print(f"Extract Non-transition samples from {self.name} dataset  :")
            print(f"Walking: {len(pid_w)} samples,  {len(set(pid_w))} unique pedestrians and {n_w} frames.")
            print(f"Standing: {len(pid_s)} samples,  {len(set(pid_s))} unique pedestrians and {n_s} frames.")

        return samples





def extract_pred_sequence(trans, non_trans=None, pred_ahead=0, balancing_ratio=None,
                          bbox_min=0, max_frames=None, seed=None, neg_in_trans=True, verbose=False) -> dict:
    """
    Extract  sequences for transition prediction task.
    :params: trans: transition history samples, i.e. GO or STOP
             non-trans: history samples containing no transitions
             pred_ahead: frame to predicted in advance, whether the trnasition occur in X frames.
             balancing_ratio: ratio between positive and negative frame instances
             bbox_min: minimum width of the pedestrian bounding box
             max_frames: maximum frames in one sequence sample
             seed: random used during balancing
             verbose: optional printing
    """
    assert isinstance(pred_ahead, int) and pred_ahead >= 0, "Invalid prediction length."
    ids_trans = list(trans.keys())
    samples = {}
    n_1 = 0
    if isinstance(bbox_min, int):
        bbox_min = (bbox_min, bbox_min)
    for idx in ids_trans:
        key_vid = trans[idx]['video_number']
        frames = copy.deepcopy(trans[idx]['frame'])
        frames_total = copy.deepcopy(trans[idx]['frame_total'])
        bbox = copy.deepcopy(trans[idx]['bbox'])
        bbox_total = copy.deepcopy(trans[idx]['bbox_total'])
        action = copy.deepcopy(trans[idx]['action'])
        action_total = copy.deepcopy(trans[idx]['action_total'])
        if "behavior" in list(trans[idx].keys()):
            behavior = copy.deepcopy(trans[idx]['behavior'])
        else:
            behavior = []
        if "attributes" in list(trans[idx].keys()):
            attributes = copy.deepcopy(trans[idx]['attributes'])
        else:
            attributes = []
        if "traffic_light" in list(trans[idx].keys()):
            traffic_light = copy.deepcopy(trans[idx]['traffic_light'])
        else:
            traffic_light = []
        d_pre = trans[idx]['pre_state']
        n_frames = len(frames)
        fps = trans[idx]['fps']
        source = trans[idx]['source']
        if source == 'TITAN':
            step = 60 // fps
        if source == 'LOKI':
            step = 10 // fps
        if source == 'PIE' or source == 'JAAD':
            step = 30 // fps
        # todo: controllare qui per TITAN, ok!. controllare per LOKI? -> forse 10 -> 10 // 5 = 2
        for i in range(max(0, n_frames - d_pre), n_frames - 1):
            if abs(bbox[i][2] - bbox[i][0]) < bbox_min[0]:
                continue
            key = idx + f"_f{frames[i]}"
            TTE = (frames[-1] - frames[i]) / (step * fps)  # TTE: questo dovrebbe essere in secondi
            if TTE > pred_ahead / fps:  # pred_ahead/fps: 2 secondi
                trans_label = 0
                key = None
                if neg_in_trans:
                    key = idx + f"_f{frames[i]}"
            else:
                trans_label = 1
                n_1 += 1
            if key is not None:

                # t = 0 if max_frames is None else i - max_frames + 1
                if max_frames is None:
                    t = 0
                else:
                    if i < max_frames - 1:
                        t = 0
                    else:
                        t = i - max_frames + 1

                if source == 'JAAD':
                    key_frame = frames[i]

                    key = f'/equilibrium/datasets/TransNet/DATA/images/JAAD/{key_vid}/' + str(key_frame).zfill(5) + '.png'
                samples[key] = {}
                samples[key]['source'] = trans[idx]['source']
                if samples[key]['source'] == 'PIE':
                    samples[key]['set_number'] = trans[idx]['set_number']
                samples[key]['video_number'] = trans[idx]['video_number']
                samples[key]['frame_current'] = frames[i]
                samples[key]['frame'] = frames[t:i + 1]  # todo: è giusto i+1? si, ok, perchè l'ultimo non lo prende
                samples[key]['bbox'] = bbox[t:i + 1]
                samples[key]['action'] = action[t:i + 1]


                samples[key]['bbox_total'] = bbox_total
                samples[key]['action_total'] = action_total
                samples[key]['frame_total'] = frames_total

                ind_fut = np.where(np.array(frames_total) == frames[i])[0].item()
                samples[key]['frame_future'] = frames_total[ind_fut + step:ind_fut + step + step*20:step]
                samples[key]['bbox_future'] = bbox_total[ind_fut + step:ind_fut + step + step*20:step]
                samples[key]['action_future'] = action_total[ind_fut + step:ind_fut + step + step*20:step]
                len_future = len(samples[key]['frame_future'])
                if len_future == 0:
                    samples[key]['frame_future'] = 0
                    samples[key]['bbox_future'] = 0
                    samples[key]['action_future'] = 0

                if len_future<20 and len_future>0:
                    samples[key]['frame_future'] = np.concatenate((samples[key]['frame_future'], np.ones(20-len(samples[key]['frame_future']))*samples[key]['frame_future'][-1])).astype(int)
                    samples[key]['bbox_future'] = np.concatenate((samples[key]['bbox_future'], np.ones((20-len(samples[key]['bbox_future']), 4))*samples[key]['bbox_future'][-1])).astype(int)
                    samples[key]['action_future'] = np.concatenate((samples[key]['action_future'], np.ones(20-len(samples[key]['action_future']))*samples[key]['action_future'][-1])).astype(int)

                if len(traffic_light) > 0:
                    samples[key]['traffic_light'] = traffic_light[t:i + 1]
                else:
                    pass
                if len(behavior) > 0:
                    samples[key]['behavior'] = behavior[t:i + 1]
                else:
                    pass
                if len(attributes) > 0:
                    samples[key]['attributes'] = attributes
                else:
                    pass
                samples[key]['trans_label'] = trans_label
                samples[key]['TTE'] = TTE
    # negative instances from all examples
    if non_trans is not None:
        action_type = 'walking' if trans[ids_trans[0]]['type'] == 'STOP' else 'standing'
        ids_non_trans = list(non_trans[action_type].keys())
        for idx in ids_non_trans:
            key_vid = non_trans[action_type][idx]['video_number']

            frames = copy.deepcopy(non_trans[action_type][idx]['frame'])
            frames_total =copy.deepcopy(non_trans[action_type][idx]['frame_total'])
            bbox = copy.deepcopy(non_trans[action_type][idx]['bbox'])
            bbox_total = copy.deepcopy(non_trans[action_type][idx]['bbox_total'])
            action = copy.deepcopy(non_trans[action_type][idx]['action'])
            action_total = copy.deepcopy(non_trans[action_type][idx]['action_total'])
            if "behavior" in list(non_trans[action_type][idx].keys()):
                behavior = copy.deepcopy(non_trans[action_type][idx]['behavior'])
            else:
                behavior = []
            if "attributes" in list(non_trans[action_type][idx].keys()):
                attributes = copy.deepcopy(non_trans[action_type][idx]['attributes'])
            else:
                attributes = []
            if "traffic_light" in list(non_trans[action_type][idx].keys()):
                traffic_light = copy.deepcopy(non_trans[action_type][idx]['traffic_light'])
            else:
                traffic_light = []
            for i in range(len(frames)):
                if abs(bbox[i][2] - bbox[i][0]) < bbox_min[1]:
                    continue
                key = idx + f"_f{frames[i]}"

                # t = 0 if max_frames is None else i - max_frames + 1
                if max_frames is None:
                    t = 0
                else:
                    if i < max_frames - 1:
                        t = 0
                    else:
                        t = i - max_frames + 1

                if source == 'JAAD':
                    key_frame = frames[i]

                    key = f'/equilibrium/datasets/TransNet/DATA/images/JAAD/{key_vid}/' + str(key_frame).zfill(5) + '.png'
                samples[key] = {}
                samples[key]['source'] = non_trans[action_type][idx]['source']
                if samples[key]['source'] == 'PIE':
                    samples[key]['set_number'] = non_trans[action_type][idx]['set_number']
                samples[key]['video_number'] = non_trans[action_type][idx]['video_number']
                samples[key]['frame'] = frames[t:i + 1]
                samples[key]['bbox'] = bbox[t:i + 1]
                samples[key]['bbox_total'] = bbox_total
                samples[key]['action_total'] = action_total
                samples[key]['frame_total'] = frames_total

                ind_fut = np.where(np.array(frames_total) == frames[i])[0].item()
                samples[key]['frame_future'] = frames_total[ind_fut + step:ind_fut + step + step*20:step]
                samples[key]['bbox_future'] = bbox_total[ind_fut + step:ind_fut + step + step*20:step]
                samples[key]['action_future'] = action_total[ind_fut + step:ind_fut + step + step*20:step]
                len_future = len(samples[key]['frame_future'])
                if len_future == 0:
                    samples[key]['frame_future'] = 0
                    samples[key]['bbox_future'] = 0
                    samples[key]['action_future'] = 0

                if len_future<20 and len_future>0:
                    samples[key]['frame_future'] = np.concatenate((samples[key]['frame_future'], np.ones(20-len(samples[key]['frame_future']))*samples[key]['frame_future'][-1])).astype(int)
                    samples[key]['bbox_future'] = np.concatenate((samples[key]['bbox_future'], np.ones((20-len(samples[key]['bbox_future']), 4))*samples[key]['bbox_future'][-1])).astype(int)
                    samples[key]['action_future'] = np.concatenate((samples[key]['action_future'], np.ones(20-len(samples[key]['action_future']))*samples[key]['action_future'][-1])).astype(int)

                samples[key]['action'] = action[t:i + 1]
                samples[key]['action_total'] = action_total
                if len(traffic_light) > 0:
                    samples[key]['traffic_light'] = traffic_light[t:i + 1]
                else:
                    pass
                if len(behavior) > 0:
                    samples[key]['behavior'] = behavior[t:i + 1]
                else:
                    pass
                if len(attributes) > 0:
                    samples[key]['attributes'] = attributes
                else:
                    pass
                samples[key]['trans_label'] = 0
                samples[key]['TTE'] = float('nan')

    if verbose:
        if n_1 > 0:
            ratio = (len(samples.keys()) - n_1) / n_1
        else:
            ratio = 999.99
        print(f'Extract {len(samples.keys())} sequence samples from {len(trans.keys())} history.')
        print('1/0 ratio:  1 : {:.2f}'.format(ratio))
        print(f'predicting-ahead frames: {pred_ahead}')

    return samples
