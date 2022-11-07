import os
import argparse
import torch
import numpy as np
from collections import defaultdict
from utils import convert_labels_to_segments

def actions_durations(vid_list, gt_path, dataset, actions_dict):
    lst_durations = []
    lst_actions = []
    for v in vid_list:
        vid = gt_path + v
        with open(vid, 'r') as f:
            content = f.read().split('\n')[:-1]
        labels = [actions_dict[a] for a in content]
        segments = convert_labels_to_segments(torch.tensor(np.array(labels)).unsqueeze(0))
        duration, action = segments['durations'][:, 1:-1], segments['labels'][:, 1:-1]
        lst_actions.append(action)
        lst_durations.append(duration)
    return lst_durations, lst_actions

def compute_mean_dur(dataset, split, data_root):
    print(f'Compute mean durations for {dataset} split {split}... ', end='')
    vid_list_file = os.path.join(data_root, f"{dataset}/splits/train.split{split}.bundle")
    gt_path = os.path.join(data_root, f"{dataset}/groundTruth/")
    mean_dur_file = os.path.join(data_root, f"{dataset}/splits/fifa_mean_dur_split{split}.pt")
    mean_length_file = os.path.join(data_root, f"{dataset}/splits/train_split{split}_mean_duration.txt")

    with open(os.path.join(data_root, f"{dataset}/mapping.txt"), 'r') as f:
        actions = f.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    with open(vid_list_file, 'r') as f:
        train = f.read().split('\n')[:-1]

    lst_durations, lst_actions = actions_durations(train, gt_path, dataset, actions_dict)

    # compute mean duration of each action
    action_duration = defaultdict(list)
    action_length = defaultdict(list)
    for action, duration in zip(lst_actions, lst_durations):
        vid_len = duration[0, :].sum()
        for a, d in zip(action[0, :], duration[0, :]):
            action_duration[a.item()].append(d / vid_len)
            action_length[a.item()].append(d)
    action_mean_dur = {}
    for action, duration in sorted(action_duration.items()):
        action_mean_dur[action] = torch.stack(duration).mean()
    action_mean_len = []
    for action, length in sorted(action_length.items()):
        action_mean_len.append(torch.stack(length).mean().round().item())

    torch.save(action_mean_dur, mean_dur_file)
    np.savetxt(mean_length_file, action_mean_len)
    mean_dur = torch.load(mean_dur_file)
    mean_len = np.loadtxt(mean_length_file)
    print('done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data', type=str, help='path to data')
    args = parser.parse_args()
    datasets = ['gtea', '50salads', 'breakfast']
    for dataset in datasets:
        if dataset == '50salads':
            splits = [1, 2, 3, 4, 5]
        else:
            splits = [1, 2, 3, 4]

        for split in splits:
            compute_mean_dur(dataset, split, args.data_root)
