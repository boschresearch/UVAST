#!/usr/bin/python2.7
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm


class DataGenerator(data.Dataset):
    def __init__(self,
                 data_root=None,
                 split=None,
                 dataset=None,
                 mode='train',
                 transform=None,
                 usebatch=False,
                 args=None,
                 len_seg_max=100,
                 features_path=None,
                 feature_type=".npy",
                 feature_mode="feature"):

        self.data_root = data_root
        self.mode = mode
        self.usebatch = usebatch
        self.transform = transform
        self.split = split
        self.dataset = dataset
        self.args = args
        self.len_seg_max = len_seg_max

        # dataset info paths
        vid_list_file = data_root+"/"+dataset+"/splits/train.split"+str(split)+".bundle"
        vid_list_file_tst = data_root+"/"+dataset+"/splits/test.split"+str(split)+".bundle"
        if features_path == None:
            features_path = data_root+"/"+dataset+"/features/"
        print("Loading features from: ", features_path)
        gt_path = data_root+"/"+dataset+"/groundTruth/"
        mapping_file = data_root+"/"+dataset+"/mapping.txt"
        # reading and mapping the names (action names) and classes (action ids)
        file_ptr = open(mapping_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        actions_dict = dict()
        # actions_dict
        for a in actions:
            actions_dict[a.split()[1]] = int(a.split()[0])
        self.actions_dict_call = actions_dict
        self.num_classes = len(actions_dict) 
        
        if mode.lower() == 'train':
            file_ptr = open(vid_list_file, 'r')
            self.list_of_examples = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            print('we have in total {} training data'.format(len(self.list_of_examples)))
            sample_rate = args.sample_rate
        elif mode.lower() == 'val':
            file_ptr = open(vid_list_file_tst, 'r')
            self.list_of_examples = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            print('we have in total {} testing data'.format(len(self.list_of_examples))) 
            sample_rate = args.sample_rate

        # ship data to dict
        self.data = dict()
        # traversing over all the data and save them into dict (self.data)
        max_len = 0
        max_len_seg = 0
        min_len = 10000000
        max_seg_dur = 0
        min_seg_dur = 10000000
        
        num_frameslist = []
        # for indv,valv in enumerate(tqdm(self.list_of_examples,desc='preparing data')):
        for indv, valv in enumerate(self.list_of_examples):
            self.data[indv] = dict()
            self.data[indv]['name'] = gt_path + valv
            if feature_type == ".npy":
                features = np.load(features_path + valv.split('.')[0] + '.npy')
            else:
                features = torch.load(features_path + valv.split(".")[0] + feature_type)[feature_mode].permute(1,0)
                file_ptr = open(gt_path + valv, 'r')
                content = file_ptr.read().split('\n')[:-1]
                features = torch.nn.functional.interpolate(features.unsqueeze(0), size=len(content)).squeeze().numpy()
            num_frames = features.shape[1]
            self.data[indv]['feat'] = features[:,::sample_rate]
            file_ptr = open(gt_path + valv, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(num_frames, len(content)),dtype=np.float32)
            for i in range(len(classes)):
                classes[i] = actions_dict[content[i]]
            self.data[indv]['label'] = classes[::sample_rate]
            self.data[indv]['label_org'] = classes  
            self.data[indv]['size'] = len(classes[::sample_rate])         
            self.data[indv]['size_org'] = len(classes)         
            max_len = max(max_len, self.data[indv]['size'])
            num_frameslist.append(num_frames)

            max_len_seg = max(max_len_seg, self.convert_labels_to_segments(torch.from_numpy(classes[::sample_rate]))['seg_gt'].shape[1])            
            min_len = min(min_len, self.data[indv]['size'])

            min_seg_dur = min(min_seg_dur, self.convert_labels_to_segments(torch.from_numpy(classes[::sample_rate]))['seg_dur'][0, 1:-1].min())
            max_seg_dur = max(max_seg_dur, self.convert_labels_to_segments(torch.from_numpy(classes[::sample_rate]))['seg_dur'][0, 1:-1].max())

        self.max_len = max_len
        print('max/min/max_seg len for this dataset in {} data is {}/{}/{}'.format(mode.lower(), max_len, min_len, max_len_seg))
        print('min_seg_dur/max_seg_dur  for this dataset in {} data is  {}/{}'.format(mode.lower(), min_seg_dur, max_seg_dur))
                          
    def __getitem__(self, index):
        feat = deepcopy(self.data[index]['feat'])
        input_tensor  = torch.from_numpy(feat)
        target_tensor = torch.from_numpy(deepcopy(self.data[index]['label']))
        target_tensor_org = torch.from_numpy(deepcopy(self.data[index]['label_org']))
        # if any augmentation is set in the arguments
        if self.args.aug_rnd_drop and self.mode=='train':
            if random.uniform(0, 1) > 0.5:
                # -- remove slight amount of frames for generalization, we ranomly drop 10 frames in the seq
                indice0 = torch.tensor([sorted(random.sample(range(input_tensor.shape[1]), input_tensor.shape[1] - random.randint(10, min(200, input_tensor.shape[1] // 2))))])[0]
                while not (torch.unique(target_tensor[indice0]).shape == torch.unique(target_tensor).shape):
                    indice0 = torch.tensor([sorted(random.sample(range(input_tensor.shape[1]), input_tensor.shape[1] - random.randint(5, 15)))])[0]
                input_tensor = input_tensor.clone()[:, indice0]
                target_tensor = target_tensor.clone()[indice0]
        mask = torch.ones_like(input_tensor)

        segments_dict = self.convert_labels_to_segments(target_tensor)
        segments_dict2 = self.convert_labels_to_segments2(target_tensor)
        len_seq = input_tensor.shape[1]
        len_seg = segments_dict['seg_gt'].shape[1]

        segments_dict_org = self.convert_labels_to_segments(target_tensor_org)

        if self.args.usebatch and self.mode == 'train':
            # we add zero to the features (right size)
            # we put -1 for class of action for those frames and
            # mask for those values would become zero
            # print(self.args.max_len)
            pad_val = self.max_len-input_tensor.shape[1]
            input_tensor = F.pad(input_tensor, (0, pad_val), mode='constant', value=0)
            target_tensor = F.pad(target_tensor, (0, pad_val), mode='constant', value=-1)
            mask = F.pad(mask, (0, pad_val), mode='constant', value=0)
            assert input_tensor.shape[-1] == self.max_len

        assert self.args.usebatch == False and self.args.bs == 1, 'you cannot use False UseBatch and BS>1'

        output_1 = {}
        output_1['feat'] = input_tensor
        output_1['gt'] = target_tensor
        output_1['gt_org'] = torch.tensor(0.0) if self.mode=='train' else target_tensor_org
        output_1['mask'] = mask
        if self.len_seg_max != 0:
            output_1['seg_gt'] = F.pad(segments_dict['seg_gt'], pad=(0, self.len_seg_max - len_seg), mode='constant', value=-1)[0] if self.mode=='train' else segments_dict['seg_gt'][0]
            output_1['seg_dur'] = F.pad(segments_dict['seg_dur'], pad=(0, self.len_seg_max - len_seg), mode='constant', value=0)[0] if self.mode=='train' else segments_dict['seg_dur'][0]
        else:
            output_1['seg_gt'] = segments_dict['seg_gt'][0]
            output_1['seg_dur'] = segments_dict['seg_dur'][0]

        output_1['seg_dur_normalized'] = F.pad(segments_dict['seg_dur'], pad=(0, self.len_seg_max - len_seg), mode='constant', value=0)[0] if self.mode=='train' else segments_dict['seg_dur'][0]
        output_1['seg_gt_org'] = torch.tensor(0.0) if self.mode == 'train' else segments_dict_org['seg_gt'][0]
        output_1['seg_dur_org'] = torch.tensor(0.0) if self.mode == 'train' else segments_dict_org['seg_dur'][0]
        output_1['seg_gt_no_split'] = segments_dict2['seg_gt'][0][1:-1]
        output_1['seg_dur_no_split'] = segments_dict2['seg_dur'][0][1:-1]
        
        output_1['len_org'] = self.data[index]['size']
        output_1['len_org_org'] = self.data[index]['size_org']
        output_1['len_seq_seg'] = (len_seq, len_seg)
        output_1['len_max_seq_seg'] = (self.max_len, self.len_seg_max)
        output_1['name'] = self.data[index]['name']
        output_1['index'] = index
        return output_1

    def __len__(self):
        return len(self.data)
    
    def convert_labels_to_segments(self, labels): # , split_segments=False, split_segments_max_dur=None
        segments = self.convert_labels(labels)
        # we need to insert <sos> and <eos>
        segments.insert(0, (torch.tensor(-2, device=labels.device), -1, -1))
        segments.append((torch.tensor(-1, device=labels.device), segments[-1][-1], segments[-1][-1]))
        if self.args.split_segments and   self.mode == 'train' and self.args.split_segments_max_dur: 
            max_dur = self.args.split_segments_max_dur # it used to be random.sample(split_segments_max_dur, 1)[0]
            segments = self.split_segments_into_chunks(segments, labels.shape[0], max_dur)
            
            
        target_labels = torch.stack([one_seg[0] for one_seg in segments]).unsqueeze(0) + 2 # two is because we are adding our sos and eos
        
        target_durations_unnormalized = self.compute_offsets([one_seg[2] for one_seg in segments]).to(target_labels.device).unsqueeze(0)
        segments_dict = {'seg_gt': target_labels,
                        'seg_dur': target_durations_unnormalized,
                        'seg_dur_normalized': target_durations_unnormalized/target_durations_unnormalized.sum().item(),
                        }
        return segments_dict
        
        
    def convert_labels_to_segments2(self, labels): # , split_segments=False, split_segments_max_dur=None
        segments = self.convert_labels(labels)
        # we need to insert <sos> and <eos>
        segments.insert(0, (torch.tensor(-2, device=labels.device), -1, -1))
        segments.append((torch.tensor(-1, device=labels.device), segments[-1][-1], segments[-1][-1]))
        
        target_labels = torch.stack([one_seg[0] for one_seg in segments]).unsqueeze(0) + 2 # two is because we are adding our sos and eos
        
        target_durations_unnormalized = self.compute_offsets([one_seg[2] for one_seg in segments]).to(target_labels.device).unsqueeze(0)
        segments_dict = {'seg_gt': target_labels,
                        'seg_dur': target_durations_unnormalized,
                        'seg_dur_normalized': target_durations_unnormalized / target_durations_unnormalized.sum().item(),
                        }
        return segments_dict
        
        
    def split_segments_into_chunks(self,segments, video_length, max_dur):
        target_durations_unnormalized = self.compute_offsets([one_seg[2] for one_seg in segments]).unsqueeze(0)

        new_segments = []
        for segment, norm_dur in zip(segments, target_durations_unnormalized[0, :] / video_length):
            if norm_dur < max_dur:
                new_segments.append(segment)
            else:
                num_chunks = int(norm_dur.item() // max_dur) + 1
                chunks = np.linspace(segment[1], segment[2] - 1, num=num_chunks + 1, dtype=int)
                start, end = chunks[:-1], chunks[1:] + 1
                for i in range(num_chunks):
                    new_segments.append((segment[0], start[i], end[i]))
        return new_segments
    

    def compute_offsets(seldf, time_stamps):
        time_stamps.insert(0, -1)
        time_stamps_unnormalized = torch.tensor([float(i - j) for i, j in zip(time_stamps[1:], time_stamps[:-1])])
        return time_stamps_unnormalized

    def convert_labels(self,labels):
        action_borders = [i for i in range(len(labels) - 1) if labels[i] != labels[i + 1]]
        action_borders.insert(0, -1)
        action_borders.append(len(labels) - 1)
        label_start_end = []
        for i in range(1, len(action_borders)):
            label, start, end = labels[action_borders[i]], action_borders[i - 1] + 1, action_borders[i]
            label_start_end.append((label, start, end))
        return label_start_end


    def start_end2center_width(self,start_end):
        return torch.stack([start_end.mean(dim=2), start_end[:,:,1] - start_end[:,:,0]], dim=2)


    def convert_segments(self,segments):
        labels = np.zeros(segments[-1][-1] + 1)
        for segment in segments:
            labels[segment[1]:segment[2] + 1] = segment[0]
        return labels    
    
