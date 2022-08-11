# code from https://github.com/TengdaHan/DPC/blob/master/utils/utils.py (MIT License) - many modifications/additions
import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import torch
import math
from eval import accuracy, edit_score, f_score
from collections import defaultdict


def get_lr_per_epoch(scheduler, num_epoch):
    lr_per_epoch = []
    for epoch in range(num_epoch):
        lr_per_epoch.append(scheduler.get_epoch_values(epoch))
    return lr_per_epoch        


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def params_count(model):
    """Compute the number of parameters."""
    return np.sum([p.numel() for p in model.parameters()]).item()


def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {} # save all data values here
        self.save_dict = {} # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count


class Metrics:
    def __init__(self):
        self.acc = AverageMeter()
        self.edit = AverageMeter()
        self.overlap = [.1, .25, .5]
        self.tp, self.fp, self.fn = np.zeros(3), np.zeros(3), np.zeros(3)
        self.tp_val, self.fp_val, self.fn_val = np.zeros(3), np.zeros(3), np.zeros(3)
        self.best_acc = (0.0, -1)
        self.best_edit = (0.0, -1)
        self.best_f1 = {.1: (0.0, -1),
                        .25: (0.0, -1),
                        .5: (0.0, -1)}

    def update_acc(self, acc, len):
        self.acc.update(acc, len)

    def update_edit(self, edit):
        self.edit.update(edit)

    def update_f1s(self, tp1, fp1, fn1, s):
        self.tp[s] += tp1
        self.fp[s] += fp1
        self.fn[s] += fn1
        self.tp_val[s] = tp1
        self.fp_val[s] = fp1
        self.fn_val[s] = fn1

    def get_current_metrics(self):
        metrics = {'Acc': self.acc.val * 100, 'Edit': self.edit.val}
        for s in range(len(self.overlap)):
            prec = np.array(np.array(self.tp_val[s]) / (self.tp_val[s] + self.fp_val[s]))
            rec = np.array(np.array(self.tp_val[s]) / (self.tp_val[s] + self.fn_val[s]))
            metrics[f'F1@{self.overlap[s]:.2f}'] = 100 * np.nan_to_num(2.0 * (prec * rec) / (prec + rec))
        return metrics

    def print(self, mode, type_model=None, epoch=0, print_best=[]):
        print(f"{mode} Acc: {self.acc.avg * 100:.4f}")
        print(f"{mode} Edit: {self.edit.avg:.4f}")
        f1s = []
        for s in range(len(self.overlap)):
            precision = self.tp[s] / float(self.tp[s] + self.fp[s])
            recall = self.tp[s] / float(self.tp[s] + self.fn[s])
            if  (precision + recall)==0:
                f1=0.0
            else:
                f1 = 2.0 * (precision * recall) / (precision + recall)
            f1 = np.nan_to_num(f1) * 100
            print(f'{mode} F1@{self.overlap[s]:0.2f}: {f1:.4f}')
            f1s.append(f1)
            if f1 > self.best_f1[self.overlap[s]][0]:
                self.best_f1[self.overlap[s]] = (f1, epoch)
        
        if self.acc.avg * 100 > self.best_acc[0]:
            self.best_acc = (self.acc.avg * 100, epoch)
        if self.edit.avg > self.best_edit[0]:
            self.best_edit = (self.edit.avg, epoch)
        if len(print_best) > 0:
            print('---------------')
            if 'acc' in print_best:
                print(f'best_acc {mode} is {self.best_acc[0]:.3f} at epoch {self.best_acc[1]}')
            if 'edit'in print_best:
                print(f'best_edit {mode} is {self.best_edit[0]:.3f} at epoch {self.best_edit[1]}')
            if 'f1@50' in print_best:
                print(f'best_f1@50 {mode} is {self.best_f1[.5][0]:.3f} at epoch {self.best_f1[.5][1]}')
        return self.acc.avg * 100, self.edit.avg, f1s


def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, 'w')
    else:
        log_file = open(filename, 'a')
    log_file.write(f'Epoch {epoch}: ')
    log_file.write(content + '\n')
    log_file.close()


def write_metrics(acc, edit, f1s, epoch, filename):
    content = f'Acc {acc:.4f} Edit {edit:.4f} F1@10 {f1s[0]:.4f} F1@25 {f1s[1]:.4f} F1@50 {f1s[2]:.4f}'
    write_log(content, epoch, filename)


def select_best_model(filename, eps=0.1):
    log_file = open(filename, 'r')
    lines = log_file.readlines()
    log_file.close()
    best_metrics = [0.0, 0.0, 0.0, 0.0, 0.0]
    best_epoch = -1
    for line in lines:
        epoch = int(line.split(":")[0][6:])
        metrics = [float(val) for val in line.split(":")[1][1:-2].split(" ")[1::2]]
        if all([cur > best - eps for cur, best in zip(metrics, best_metrics)]):
            best_metrics = metrics
            best_epoch = epoch
    print(f'Best model at Epoch {best_epoch} with Acc {best_metrics[0]} Edit {best_metrics[1]} F1@10 {best_metrics[2]} F1@25 {best_metrics[3]} F1@50 {best_metrics[4]}')
    return best_metrics


def start_end2center_width(start_end):
    return torch.stack([start_end.mean(dim=2), start_end[:,:,1] - start_end[:,:,0]], dim=2)


def convert_segments(segments):
    labels = np.zeros(segments[-1][-1] + 1)
    for segment in segments:
        labels[segment[1]:segment[2] + 1] = segment[0]
    return labels


def convert_labels(labels):
    action_borders = [i for i in range(len(labels) - 1) if labels[i] != labels[i + 1]]
    action_borders.insert(0, -1)
    action_borders.append(len(labels) - 1)
    label_start_end = []
    for i in range(1, len(action_borders)):
        label, start, end = labels[action_borders[i]], action_borders[i - 1] + 1, action_borders[i]
        label_start_end.append((label, start, end))
    return label_start_end
    

def update_metrics(recognition, gt_cls, metrics):
    metrics.update_acc(accuracy(recognition, gt_cls), len(recognition))
    metrics.update_edit(edit_score(recognition, gt_cls))
    for s in range(len(metrics.overlap)):
        tp1, fp1, fn1 = f_score(recognition, gt_cls, metrics.overlap[s])
        metrics.update_f1s(tp1, fp1, fn1, s)


def convert_labels_to_segments(labels):
    bs = labels.shape[0]
    assert bs == 1, 'Not yet implemented for larger batchsizes.'
    labels = labels[0, :]
    segments = convert_labels(labels)
    # we need to insert <sos> and <eos>
    segments.insert(0, (torch.tensor(-2, device=labels.device), -1, -1))
    segments.append((torch.tensor(-1, device=labels.device), segments[-1][-1], segments[-1][-1]))
    target_labels = torch.stack([s[0] for s in segments]).unsqueeze(0) + 2
    start_end = torch.stack([torch.tensor([s[1], s[2]]) for s in segments]).unsqueeze(0).float()
    center_width = start_end2center_width(start_end)
    #start_end_norm = start_end / start_end[:,-1,-1]
    #center_width_norm = start_end2center_width(start_end_norm)
    target_durations_unnormalized = compute_offsets([s[2] for s in segments]).to(target_labels.device).unsqueeze(0)
    segments_dict = {'labels': target_labels,
                     'durations': target_durations_unnormalized,
                     'start_end': start_end.to(target_labels.device),
                     'center_width': center_width.to(target_labels.device)}
    return segments_dict


def compute_offsets(time_stamps):
    #bs = time_stamps.shape[0]
    #assert bs == 1, 'Not yet implemented for larger batchsizes.'
    time_stamps.insert(0, -1)
    time_stamps_unnormalized = torch.tensor([float(i - j) for i, j in zip(time_stamps[1:], time_stamps[:-1])])
    return time_stamps_unnormalized

def convert_segments_to_labels(action, duration, num_frames, args, do_quantization=None):
    bs = action.shape[0]
    assert bs == 1, 'Not yet implemented for larger batchsizes.'
    if do_quantization == 'quantize':
        labels = action[0, :] - 2
        frame_wise_predictions = torch.zeros((1, num_frames))
        idx = 0
        for i in range(labels.shape[0]):
            q_spans = torch.arange(0,10000,args.quantization_span).to(labels.device)
            segm_dur = int((duration[i]*q_spans).sum().item())
            if int(idx + segm_dur)<=num_frames:
                frame_wise_predictions[0, idx:int(idx + segm_dur)] = labels[i]
                idx += int(segm_dur)
            else:
                if int(idx)<=num_frames:
                    frame_wise_predictions[0, idx:] = labels[i]
                    idx += int(segm_dur)
        frame_wise_predictions = frame_wise_predictions[:num_frames]
    else:
        labels = action[0, :] - 2
        duration = duration[0, :]
        duration = duration / duration.sum()
        #num_frames = input_x.shape[-1]
        duration = (duration * num_frames).round().long()
        if duration.shape[0] == 0:
            duration = torch.tensor([num_frames])
            labels = torch.tensor([0])
        if duration.sum().item() != num_frames:
            # there may be small inconsistencies due to rounding.
            duration[-1] = num_frames - duration[:-1].sum()
        assert duration.sum().item() == num_frames, f"Prediction {duration.sum().item()} does not match number of frames {num_frames}."
        frame_wise_predictions = torch.zeros((1, num_frames))
        idx = 0
        for i in range(labels.shape[0]):
            frame_wise_predictions[0, idx:idx + duration[i]] = labels[i]
            idx += duration[i]
    return frame_wise_predictions


def framewise_duration(duration_normalized, duration_unnormalized):
    time_stamps = torch.cumsum(duration_unnormalized, dim=1).long()[:, :-1] + 1
    time_stamps[:, 0] = 0
    framewise_duration = torch.zeros((1, int(time_stamps[0, -1].item()))).to(duration_normalized.device)
    for start, end, val in zip(time_stamps[0, :-1], time_stamps[0, 1:], duration_normalized[0, 1:]):
        framewise_duration[0, start:end] = val
    return framewise_duration


def GIoU(recognized, ground_truth):
    rec_start = recognized[:-1]
    rec_end = recognized[1:]
    gt_start = ground_truth[:-1]
    gt_end = ground_truth[1:]
    intersection = torch.clamp(torch.min(rec_end - gt_start, gt_end - rec_start), min=0)
    union = torch.max(rec_end - gt_start, gt_end - rec_start)
    empty_space = torch.clamp(torch.max(gt_start - rec_end, rec_start - gt_end), min=0)
    convex_hull = torch.max(gt_end - rec_start, rec_end - gt_start)
    generalizedIoU = intersection / union - empty_space / convex_hull
    return generalizedIoU


class CurriculumStrategy:
    def __init__(self, strategy, num_epochs):
        self.strategy = strategy
        self.num_epochs = num_epochs

    def __call__(self, epoch):
        if self.strategy is None:
            curriculum = None
        elif self.strategy == 'linear':
            curriculum = epoch / (self.num_epochs - 1)
        return curriculum


def simplify_durations(labels, features, downsample=False):
    is_np = isinstance(labels, np.ndarray)
    if is_np:
        labels = torch.tensor(labels).unsqueeze(0).long()
        features = torch.tensor(features).unsqueeze(0)
    segments = convert_labels_to_segments(labels)
    if downsample:
        len = 1
    else:
        len = 100
    simple_durations = (segments['durations'] // 100) * len
    n_frames = simple_durations.sum().long().item()
    new_labels = convert_segments_to_labels(segments['labels'], simple_durations, n_frames, args=None, do_quantization=None).long()
    f = torch.zeros(1, features.shape[1], n_frames)
    idx = 0
    idx_f = 0
    for d, df in zip(simple_durations[0, :], segments['durations'][0, :]):
        offset = d.long().item()
        f[:, :, idx:idx+offset] = features[:, :, idx_f:idx_f+offset]
        idx += offset
        idx_f += df.long().item()
    if is_np:
        new_labels = new_labels[0, :].numpy()
        f = f[0, :].numpy()
    return new_labels, f


def augment_segments(features, classes, stretch=False, max_seg_length=None, permute=False):
    segments = convert_labels(classes)
    features_segments = []
    for segment in segments:
        features_segments.append(features[:, segment[1]:segment[2] + 1])
    idx = 0
    new_segments = []
    new_features = []
    shapes = torch.tensor([f.shape[1] for f in features_segments])
    new_lens = shapes
    if max_seg_length:
        exceed_len = shapes > max_seg_length
        new_lens[exceed_len] = ((3 / 4 + (1 / 4) * torch.rand(len(shapes[exceed_len]))) * max_seg_length).long()
    elif stretch:
        new_lens = ((3 / 4 + (4 / 3 - 3 / 4) * torch.rand(len(shapes))) * shapes).long()
        while new_lens.sum() > 9750:
            new_lens = ((3 / 4 + (4 / 3 - 3 / 4) * torch.rand(len(shapes))) * shapes).long()
    if permute:
        permutation = torch.randperm(len(features_segments) - 2)
        new_lens[1:-1] = new_lens[1:-1][permutation]
        tmp = []
        tmp.append(features_segments[0])
        for p in permutation:
            tmp.append(features_segments[p.item() + 1])
        tmp.append(features_segments[-1])
        features_segments = tmp
    for i in range(len(features_segments)):
        new_len = new_lens[i].item()
        new_segments.append((segments[i][0], idx, idx + new_len - 1))
        idx += new_len
        if features_segments[i].shape[1] == new_len:
            new_features.append(features_segments[i])
        else:
            new_features.append(
                torch.nn.functional.interpolate(torch.tensor(features_segments[i]).unsqueeze(0), size=new_len)[0, :,
                :].numpy())
    new_classes = convert_segments(new_segments)
    new_features = np.concatenate(new_features, axis=1)
    return new_features, new_classes


def _freeze_norm_stats(net):
    try:
        for m in net.modules():
            if isinstance(m, nn.InstanceNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LazyBatchNorm1d) or isinstance(m, nn.LazyBatchNorm2d):
                # m.track_running_stats = False
                m.eval()
    except ValueError:  
        print("errrrrrrrrrrrrrroooooooorrrrrrrrrrrr with instancenorm")
        return


def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm1d') != -1:
              m.eval()
              
             
def refine_transcript(transcript):
    transcript_refined = []
    for i, action in enumerate(transcript):
        if i == 0:
            transcript_refined.append(action.item())
        elif i != 0 and action != transcript[i - 1]:
            transcript_refined.append(action.item())

    transcript_refined = F.pad(torch.tensor(transcript_refined).unsqueeze(0), pad=(1, 1))
    return transcript_refined

def remove_duplicates_from_transcript(pred_transcript, out_dec):
    dictt = defaultdict(list)
    for kkind, valll in enumerate(pred_transcript[0]):
        dictt[valll.item()].append(out_dec[-1][kkind])

    pred_transcript_no_rep = []
    dec_feat = []
    for i, action in enumerate(pred_transcript[0, :]):
        if len(pred_transcript_no_rep) == 0 or pred_transcript_no_rep[-1] != action:
            pred_transcript_no_rep.append(action.unsqueeze(0))
            #dec_feat.append(out_dec[-1][i].unsqueeze(0)) # this looks correct to me, the other one is overwriting action segments!
            dec_feat.append(dictt[action.item()][0].unsqueeze(0))
    pred_transcript_no_rep = torch.cat(pred_transcript_no_rep, dim=0).unsqueeze(0)
    dec_feat = torch.cat(dec_feat, dim=0).unsqueeze(0)
    return pred_transcript_no_rep, dec_feat
