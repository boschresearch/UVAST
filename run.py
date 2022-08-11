#!/usr/bin/python2.7
# code from https://github.com/yabufarha/ms-tcn (MIT+CC License) - many modifications
import argparse
import datetime
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytz
import torch

from dataloder import DataGenerator as DataGenerator
from model import Trainer

parser = argparse.ArgumentParser()
# general arguments
parser.add_argument('--features_dim', default='2048', type=int, help='dimension of input features (I3D features are 2048 dim)')
parser.add_argument('--experiment_path', default='/path_to_directory/',  type=str, help='directory to save experiment results')
# dataset and dataloader
parser.add_argument('--split', default='1', type=int, help='split of the dataset')
parser.add_argument('--data_root', default='/path_to_data_directory/',  type=str, help='root to the datasets directory')
parser.add_argument('--data_root_mean_duration', default='/path_to_data_directory/',  type=str, help='root to the mean durations of the datasets')
parser.add_argument('--features_path', default=None, type=str)
parser.add_argument('--feature_type', default=".npy", type=str)
parser.add_argument('--feature_mode', default='feature', type=str)
parser.add_argument('--num_classes', default=-1, type=int, help='number of classes for each dataset (gtea=11, salads=19, breakfast=48)')
parser.add_argument('--skip_inference', default=False, action='store_true', help='skip inference')
parser.add_argument('--do_timing', default=False, action='store_true', help='do time measuring')
parser.add_argument('--dataset', default=None, type=str)
parser.add_argument('--sample_rate', default=1, type=int, help='frame sampling rate (salad=2, beakfast,gtea=1)')
parser.add_argument('--aug_rnd_drop', default=False, action='store_true', help='optional augmention for training: randomly dropping frames')
parser.add_argument('--split_segments', default=False, action='store_true', help='splitting segments')
parser.add_argument('--split_segments_max_dur', default='0.1', type=float, help='max duration in split_segments; for details see the paper')
parser.add_argument('--use_cuda', action="store_true",help='use cuda for training')
parser.add_argument('--exp_name', default='choose_A_NAME', type=str,help='name of the experiment, specifies the folder where models are saved')
parser.add_argument('--seed',  default=None, type=int, help='specify seed number')
parser.add_argument('--save_args', default=False, action='store_true', help='save arguments that were used for training')
# testing
parser.add_argument('--inference_only', default=False, action='store_true', help='run inference only')
parser.add_argument('--path_inference_model', default=None, type=str, help='path to model for inference')
parser.add_argument('--epoch_num_for_testing', default=-1, type=int,help='evaluate specific epoch')
parser.add_argument('--use_fifa', action='store_true', default=False, help='use fifa during inference')
parser.add_argument('--fifa_init_dur', action='store_true', default=False, help='initialize durations in fifa with alignment decoder predictions')
parser.add_argument('--use_viterbi', action='store_true', default=False, help='use viterbi during inference')
parser.add_argument('--viterbi_sample_rate', default=-1, type=int, help='sampling rate used in viterbi')
parser.add_argument('--do_refine_actions_no_consecutive_similar_action', action='store_true', default=False, help='remove duplicate actions during viterbi inference')
# encoder arguments
parser.add_argument('--num_f_maps', default='64', type=int, help='feature dimension in the transformer')
parser.add_argument('--activation', default='gelu', type=str, help='type of activation used in the enc and dec')
parser.add_argument('--encoder_model', default='asformer_advanced', type=str, choices=['asformer_advanced', 'asformer_org_enc', 'asformer_org_enc_dec'], help='select encoder model')
parser.add_argument('--enc_norm_type', default='InstanceNorm1d', type=str, help=['LayerNorm', 'InstanceNorm1d_track', 'InstanceNorm1d'])
parser.add_argument('--num_layers_enc',  default=10, type=int, help='the number of encoder layers in ASFormer')
parser.add_argument('--num_layers_asformer_dec',  default=10, type=int, help='number of decoder layers in the ASFormer (only used with asformer_org_enc_dec)')
# transcript decoder arguments
parser.add_argument('--n_head_dec', default=1, type=int, help='the number of heads of the transcript decoder (first stage)')
parser.add_argument('--AttentionPoolType_dec', default='none', type=str,help='option to smooth the cross attention; options are: none, avg, max')
parser.add_argument('--AttentionPoolKernel_dec', default=31, type=int,help='kernel size for smoothing if --AttentionPoolType_dec is selected')
parser.add_argument('--num_layers_dec', default=2, type=int,help='number of layers of the transcript decoder (first stage)')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout for the transcript decoder')
parser.add_argument('--use_pe_tgt', default=False, action='store_true', help='use positional encoding for target in the transcript decoder')
parser.add_argument('--use_pe_src', default=False, action='store_true', help='use positional encoding for source in the transcript decoder')
parser.add_argument('--dec_dim_feedforward', default=2048, type=int,help='similar to dec_dim_feedforward option in the standard pytorch model')
parser.add_argument('--len_seg_max',  default=50, type=int, help='the max number of action segments')
# alignment decoder arguments
parser.add_argument('--use_alignment_dec', default=False, action='store_true', help='use alignment decoder for duration prediction (second stage)')
parser.add_argument('--n_head_dec_dur_uvast', default=1, type=int, help='the number of heads of the alignment decoder')
parser.add_argument('--add_tgt_pe_dec_dur', default=0, type=float, help='set to 1 to add pe in the alignment decoder')
parser.add_argument('--AttentionPoolType_dec_dur', default='none', type=str,help='option to smooth the cross attention; options are: none, avg, max')
parser.add_argument('--AttentionPoolKernel_dec_dur', default=0, type=int,help='kernel size for smoothing if --AttentionPoolType_dec_dur is selected')
parser.add_argument('--dropout_dec_dur', default=0.1, type=float, help='dropout for the alignment decoder')
parser.add_argument('--dec_dim_feedforward_dec_dur', default=1024, type=int,help='similar to dec_dim_feedforward option in the standard pytorch model')
parser.add_argument('--num_layers_dec_dur', default=1, type=int,help='number of layers for the alignment decoder')
parser.add_argument('--alignment_decoder_model', default='uvast_decoder', type=str, choices=['uvast_decoder', 'pytorch_decoder'], help='select alignment decoder model')
parser.add_argument('--pretrained_model', default=None, type=str, help='path to pretrained model from first stage')
# optimizer arguments
parser.add_argument('--bs', default='1', type=int,help='batch size')
parser.add_argument('--usebatch', default=False, action='store_true', help='if bs > 1 is used the samples need to be padded')
parser.add_argument('--num_epochs', default=600, type=int, help='number of epochs')
parser.add_argument('--step_size', default='400', type=int,help='the epoch when the lr will drop by factor of 10')
parser.add_argument('--lr_scheduler', default=False, action='store_true', help='use reduce on plateau lr scheduler')
parser.add_argument('--learning_rate', default=0.0005, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float, help='gamma for the optimizer')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--optimizer', default='adam', type=str, help='optimizer type')
parser.add_argument('--adap_clip_gradient', default=False, action='store_true', help='use gradient clipping')
parser.add_argument('--clip_percentile', default=10, type=int, help='adaptive gradient clipping is based on percentile of the previous gradient')
# loss arguments
parser.add_argument('--temperature', default=0.001, type=float, help='the temperature in the cross attention loss')
parser.add_argument('--do_framewise_loss', default=False, action='store_true', help='use frame wise loss after encoder')
parser.add_argument('--do_framewise_loss_g', default=False, action='store_true', help='use group wise frame wise loss')
parser.add_argument('--do_segwise_loss', default=False, action='store_true', help='use segment wise ce loss')
parser.add_argument('--do_segwise_loss_g', default=False, action='store_true', help='use group segment wise ce loss')
parser.add_argument('--do_crossattention_action_loss_nll', default=False, action='store_true', help='use cross attention loss for first stage')
parser.add_argument('--do_crossattention_dur_loss_ce', default=False, action='store_true', help='use cross attention loss for second stage')
parser.add_argument('--framewise_loss_g_apply_logsoftmax', default=False, action='store_true', help='type of the normalization for group wise CE') 
parser.add_argument('--framewise_loss_g_apply_nothing', default=False, action='store_true', help='type of the normalization for group wise CE-this is normal averaging')
parser.add_argument('--segwise_loss_g_apply_logsoftmax', default=False, action='store_true', help='type of the normalization for group wise CE')  
parser.add_argument('--segwise_loss_g_apply_nothing', default=False, action='store_true', help='type of the normalization for group wise CE-this is normal averaging') 
args = parser.parse_args()
# #############################################################################################################################################################################################


device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.dataset == 'gtea':
    args.num_classes = 11
    args.sample_rate = 1
    args.channel_masking_rate = 0.5
    args.split_segments_max_dur = 0.17
elif args.dataset == '50salads':
    args.num_classes = 19
    args.sample_rate = 2
    args.channel_masking_rate = 0.3
    args.split_segments_max_dur = 0.15
elif args.dataset == 'breakfast':
    args.num_classes = 48
    args.sample_rate = 1
    args.channel_masking_rate = 0.3
    args.split_segments_max_dur = 0.17

if args.seed:
    print('seed is set to {}'.format(args.seed))
    # setting the seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

print(subprocess.check_output(["conda", "--version"]))
print('pytorch version {}', torch.__version__)
print('numpy version {}', np.__version__)
print('cuda', torch.version.cuda)
print('cudnn', torch.backends.cudnn.version())
print('device is:', device)


if args.exp_name:
    args.results_path = os.path.join(os.path.dirname(args.experiment_path), args.exp_name, 'results')
else:
    print('Oops!! You did not provide exp_name and it can cause your experiments results be saved in one folder and cause issues later!')
    args.results_path = os.path.join(os.path.dirname(args.experiment_path), 'results')

model_dir = os.path.join(args.results_path.replace('results', 'model'), args.dataset, 'split_' + str(args.split))
results_dir = os.path.join(args.results_path, args.dataset, 'split_' + str(args.split))
args.results_dir = results_dir
if not os.path.exists(model_dir) and not args.inference_only:
    os.makedirs(model_dir)
if not os.path.exists(results_dir) and not args.inference_only:
    os.makedirs(results_dir)

print('results_dir', results_dir)
print('model_dir', model_dir)
args.model_dir = model_dir

#print('start time:')
#print(datetime.datetime.now(pytz.timezone('US/Eastern')).isoformat())

# saving args to a file
if args.save_args and not args.inference_only:
    json_name = os.path.join(model_dir, 'args.txt')
    with open(json_name, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

args.device = device

print('data processing:')
training_dataloader = DataGenerator(data_root=args.data_root, split=args.split,
                                    dataset=args.dataset, mode='train',
                                    args=args, usebatch=args.usebatch, len_seg_max=args.len_seg_max,
                                    features_path=args.features_path, feature_type=args.feature_type, feature_mode=args.feature_mode)
trainloader = torch.utils.data.DataLoader(training_dataloader,
                                          batch_size=args.bs,
                                          shuffle=True,
                                          num_workers=4 ,
                                          pin_memory=True)  
# dataloaders testing                    
testing_dataloader = DataGenerator(data_root=args.data_root, split=args.split,
                                    dataset=args.dataset, mode='val',args=args,
                                    features_path=args.features_path, feature_type=args.feature_type, feature_mode=args.feature_mode)
testloader = torch.utils.data.DataLoader(testing_dataloader,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=4,
                                         pin_memory=True)             
print('starting the training:')
trainer = Trainer(args)
trainer.train(args, device=device, trainloader=trainloader, testloader=testloader, testing_dataloader=testing_dataloader)

#print('end time:')
#print(datetime.datetime.now(pytz.timezone('US/Eastern')).isoformat())
