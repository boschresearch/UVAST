#!/usr/bin/python2.7
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from eval import update_metrics
from losses import AttentionLoss, DurAttnCALoss, FrameWiseLoss, SegmentLossAction
from transformers_models import uvast_model
from utils import Metrics, get_grad_norm, params_count, write_metrics, refine_transcript

from FIFA import fifa
from viterbi import Viterbi, PoissonModel

grad_history=[]        
class Trainer:
    def __init__(self, args):
        self.model = uvast_model(args)
        self.args = args
        
        print('params count:', params_count(self.model))

        # initialize losses
        self.frame_wise_loss = FrameWiseLoss(args)
        self.segment_wise_loss = SegmentLossAction(args)
        self.attn_action_loss = AttentionLoss(args)
        self.attn_dur_loss = DurAttnCALoss(args)
        self.model_dir = args.model_dir


    def train(self, args, device=None, trainloader=None, testloader=None, testing_dataloader=None):
        if args.use_alignment_dec and not args.inference_only:
            self.model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'), strict=False)

            for name, p in self.model.enc_feat.named_parameters():
                p.requires_grad = False
            for name, p in self.model.dec_action.named_parameters():
                p.requires_grad = False
            for name, p in self.model.prediction_action.named_parameters():
                p.requires_grad = False
            for name, p in self.model.pos_embed.named_parameters():
                p.requires_grad = False
            for name, p in self.model.dec_embedding.named_parameters():
                p.requires_grad = False

        self.model.train()
        self.model.to(device)
        if not self.args.inference_only:
            if args.optimizer == 'adam':
                optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=self.args.weight_decay)    
            
            elif args.optimizer == 'adamw':
                optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=self.args.weight_decay)

            lr_scheduler = None
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma, verbose=True)
            if self.args.lr_scheduler:
                lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, mode="min", verbose=True)
                print("use ReduceLSOnPlateau scheduler")

        for epoch in range(args.num_epochs):
            print('epoch', epoch, flush=True)
            if args.inference_only:
                self.inference(testing_dataloader, testloader, epoch + 1, device)
                break
            start = time.process_time() 
            epoch_loss = self.train_one_epoch(trainloader, optimizer, epoch, device)
            end = time.process_time()
            if args.do_timing:
                print('time:', end - start)

            if not args.skip_inference:
                self.inference(testing_dataloader, testloader, epoch + 1, device)
            
            if self.args.lr_scheduler:
                lr_scheduler.step(epoch_loss)
            else:
                lr_scheduler.step()  


    def train_one_epoch(self, trainloader, optimizer, epoch, device):
        self.model.train()
        self.frame_wise_loss.reset()
        self.segment_wise_loss.reset()
        self.attn_action_loss.reset()
        self.attn_dur_loss.reset()

        epoch_loss = 0
        clip_gradient_value = 0.0
        optimizer.zero_grad()
        
        for index, data in enumerate(trainloader):
            framewise_losses = torch.tensor(0.0)
            segwise_losses = torch.tensor(0.0)
            attn_action_losses = torch.tensor(0.0)
            attn_duration_losses = torch.tensor(0.0)

            self.model.zero_grad(set_to_none=True)
            optimizer.zero_grad() 

            feat, gt, mask, seg_gt, seg_dur = data['feat'], data['gt'], data['mask'], data['seg_gt'], data['seg_dur']

            # normalizing the durations
            seg_dur_norm = seg_dur / data['len_seq_seg'][0].unsqueeze(1)
            batch_input, batch_target, mask = feat.to(device), gt.to(device), mask.to(device)
            seg_gt_act, seg_dur_normalize = seg_gt.to(device), seg_dur_norm.to(device)
            
            # to ignore the <eos> token we change it to -1 (in act class) and zero (in duration)
            seg_gt_act_train = seg_gt_act.clone()
            seg_dur_normalize_train = seg_dur_normalize.clone()
            mask_train = mask.clone()
            seg_gt_act_train[seg_gt_act == 1] = -1
            seg_dur_normalize_train[seg_gt_act == 1] = 0
            seg_data = (seg_gt_act_train, seg_dur_normalize_train)
            
            seg_gt_no_split = data['seg_gt_no_split'].to(device) 
            seg_dur_no_split = data['seg_dur_no_split'].to(device) 

            predictions_framewise, pred_transcript, pred_crossattn, frames_to_segment_assignment = self.model(batch_input, mask_train, seg_data, no_split_data=(seg_gt_no_split, seg_dur_no_split))

            # generarting gt for attntion maps
            attn_mask_gt = torch.zeros(seg_dur_no_split.shape[0], seg_dur_no_split.shape[1], int(seg_dur_no_split.sum().item())) 
            seg_cumsum = torch.cumsum(seg_dur_no_split, dim=1)
            for i in range(seg_dur_no_split.shape[1]):
                if i > 0:
                    attn_mask_gt[0, i, int(seg_cumsum[0, i - 1].item()):int(seg_cumsum[0, i].item())] = 1
                else:
                    attn_mask_gt[0, i, :int(seg_cumsum[0, i].item())] = 1
            attn_mask_gt_dur = attn_mask_gt.to(batch_input.device)
            
            # apply losses
            if self.args.do_framewise_loss or self.args.do_framewise_loss_g:
                framewise_losses = self.frame_wise_loss(predictions=predictions_framewise, batch_target=batch_target, mask=mask, epoch=epoch)

            if self.args.do_segwise_loss or self.args.do_segwise_loss_g:
                seg_gt_act_loss = F.pad(seg_gt_act.clone()[:, 1:], pad=(0, 1), mode='constant', value=-1)
                segwise_losses = self.segment_wise_loss(pred_transcript, seg_gt_act_loss, batch_input.shape[-1], epoch)

            if self.args.do_crossattention_action_loss_nll:    
                attn_action_losses = self.attn_action_loss(pred_crossattn, attn_mask_gt, batch_target)
                        
            if self.args.use_alignment_dec and self.args.do_crossattention_dur_loss_ce:
                attn_duration_losses = self.attn_dur_loss(frames_to_segment_assignment, attn_mask_gt_dur) 
                
            loss = framewise_losses + segwise_losses + attn_action_losses + attn_duration_losses
            loss.backward()

            if self.args.adap_clip_gradient:
                obs_grad_norm = get_grad_norm(self.model)
                grad_history.append(obs_grad_norm)
                clip_gradient_value = max(np.percentile(grad_history, self.args.clip_percentile), 0.1)
                if clip_gradient_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_gradient_value)
                    
            optimizer.step()
            
            epoch_loss += loss.item()
            
        torch.save(self.model.state_dict(), self.model_dir + "/epoch-" + str(epoch + 1) + ".model")
        torch.save(optimizer.state_dict(), self.model_dir + "/epoch-" + str(epoch + 1) + ".opt")
            
        # log training losses 
        if self.args.do_framewise_loss or self.args.do_framewise_loss_g:
            self.frame_wise_loss.log_metrics(mode="train_framewise", epoch=epoch + 1)
        if self.args.do_segwise_loss or self.args.do_segwise_loss_g:
            self.segment_wise_loss.log_metrics(mode="segment_framewise", epoch=epoch + 1)
        if self.args.do_crossattention_action_loss_nll: 
            self.attn_action_loss.log_metrics(mode="crossattention action", epoch=epoch + 1)
        if self.args.do_crossattention_dur_loss_ce:
            self.attn_dur_loss.log_metrics(mode="crossattention duration", epoch=epoch + 1)

        if self.args.adap_clip_gradient:
            print('clip_gradient {:.3f}'.format(clip_gradient_value))
        return epoch_loss        
        

    def inference(self, testing_dataloader, testloader, epoch, device):
        actions_dict_inv = {v: k for k, v in testing_dataloader.actions_dict_call.items()}
        self.model.eval()

        metrics_framewise = Metrics()
        metrics_segmentwise = Metrics()
        metrics_segmentwise_dur = Metrics()
        if self.args.use_fifa:
            metrics_seg_fifa = Metrics()
        if self.args.use_viterbi:
            metrics_seg_viterbi = Metrics()

        with torch.no_grad():
            self.model.to(device)
            if self.args.inference_only:
                if self.args.path_inference_model:
                    path_to_model = self.args.path_inference_model
                    self.args.results_dir = os.path.dirname(path_to_model)
                else:
                    pretrained_name_for_testing = self.args.exp_name
                    path_to_model = self.args.experiment_path + pretrained_name_for_testing + \
                                '/model/' + self.args.dataset + '/split_' + str(self.args.split) + '/epoch-' + str(self.args.epoch_num_for_testing) + '.model'
                print('LOADING the model {}'.format(path_to_model))

                self.model.load_state_dict(torch.load(path_to_model, map_location="cpu"), strict=True)
                
            for index, data in enumerate(tqdm(testloader, desc="testing epoch {}".format(epoch), leave=False)):
                feat, gt, gt_org, mask = data['feat'], data['gt'], data['gt_org'], data['mask'] 
                batch_input, batch_target, batch_target_org, mask = feat.to(device), gt.to(device), gt_org.to(device), mask.to(device)
                
                gt_cls_names = []
                for i in range(gt_org.shape[1]):
                    gt_cls_names.extend([actions_dict_inv[batch_target_org[:, i].item()]])

                pred_framewise, pred_transcript, pred_dur, pred_dur_AD, pred_transcript_AD = self.model(batch_input, mask)

                assert batch_input.shape[0] == 1 # we only evaluate one sample at a time
                
                # framewise predictions
                _, predicted_framewise = torch.max(pred_framewise[-1].data, 1)
                recog = self.convert_id_to_actions(predicted_framewise, gt_org, actions_dict_inv)
                update_metrics(recog, gt_cls_names, metrics_framewise)

                # transcript predictions
                pred_seg_expanded = self.convert_segments_to_labels(pred_transcript, pred_dur, feat.shape[-1])
                pred_seg_expanded = torch.clamp(pred_seg_expanded, min=0, max=self.args.num_classes)
                recog_seg = self.convert_id_to_actions(pred_seg_expanded, gt_org, actions_dict_inv)
                update_metrics(recog_seg, gt_cls_names, metrics_segmentwise)

                # alignment decoder predictions
                if self.args.use_alignment_dec:
                    assert pred_transcript_AD.shape == pred_dur_AD.shape
                    pred_seg_expanded_dur = self.convert_segments_to_labels(pred_transcript_AD, pred_dur_AD, feat.shape[-1])
                    pred_seg_expanded_dur = torch.clamp(pred_seg_expanded_dur, min=0, max=self.args.num_classes)
                    recog_seg_dur = self.convert_id_to_actions(pred_seg_expanded_dur, gt_org, actions_dict_inv)
                    update_metrics(recog_seg_dur, gt_cls_names, metrics_segmentwise_dur)
                
                # evaluation with Viterbi
                if self.args.use_viterbi:
                    viterbi = Viterbi(self.args, frame_sampling=self.args.viterbi_sample_rate, max_hypotheses=np.inf)
                    frame_probs = torch.softmax(pred_framewise[-1].detach(), dim=1)[0, :, :].transpose(0, 1)
                    # refine predicted transcript: remove repeating actions
                    transcript = refine_transcript(pred_transcript[0]).to(self.args.device)
                    duration_viterbi = viterbi.decode(frame_probs, transcript, self.args.num_classes)
                    pred_seg_expanded_dur = self.convert_segments_to_labels(transcript[:, 1:-1], duration_viterbi, feat.shape[-1])
                    pred_seg_expanded_dur = torch.clamp(pred_seg_expanded_dur, min=0, max=self.args.num_classes)
                    recog_seg = self.convert_id_to_actions(pred_seg_expanded_dur, gt_org, actions_dict_inv)
                    update_metrics(recog_seg, gt_cls_names, metrics_seg_viterbi)
                                        
                # evaluation with FIFA
                if self.args.use_fifa:
                    if self.args.fifa_init_dur:
                        actions = pred_transcript_AD.detach()
                        durations = pred_dur_AD.detach()
                        
                    duration_fifa = fifa(action=actions, duration=durations, framewise_pred=pred_framewise[-1].detach(),
                                         device=device, args=self.args)
                    pred_seg_expanded_dur = self.convert_segments_to_labels(actions, duration_fifa.detach(), feat.shape[-1])
                    pred_seg_expanded_dur = torch.clamp(pred_seg_expanded_dur, min=0, max=self.args.num_classes)
                    recog_seg = self.convert_id_to_actions(pred_seg_expanded_dur, gt_org, actions_dict_inv)
                    update_metrics(recog_seg, gt_cls_names, metrics_seg_fifa)

            mode = 'val'
            print(f'performance at epoch {epoch}')
            print('---------------')
            metrics_framewise.print(mode + " framewise", epoch=epoch, print_best=['acc'])
            print('---------------')
            metrics_segmentwise.print(mode + " segmentwise", epoch=epoch, print_best=['edit'])
                
            print('---------------')
            if self.args.use_alignment_dec:
                acc, edit, f1s = metrics_segmentwise_dur.print(mode + " segmentwise dur", epoch=epoch, print_best=['acc', 'edit', 'f1@50'])
                dir_name = os.path.join(self.args.results_dir, "AD")
                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)
                write_metrics(acc, edit, f1s, epoch, os.path.join(dir_name, "metrics.md"))

            if self.args.use_fifa:
                print('---------------')
                acc, edit, f1s = metrics_seg_fifa.print(mode + " fifa", epoch=epoch)
                dir_name = os.path.join(self.args.results_dir, "fifa")
                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)
                write_metrics(acc, edit, f1s, epoch, os.path.join(dir_name, "metrics.md"))
                
            if self.args.use_viterbi:
                print('---------------')
                acc, edit, f1s = metrics_seg_viterbi.print(mode + " viterbi", epoch=epoch)
                dir_name = os.path.join(self.args.results_dir, "viterbi")
                if not os.path.isdir(dir_name):
                    os.makedirs(dir_name)
                write_metrics(acc, edit, f1s, epoch, os.path.join(dir_name, "metrics.md"))
                
                
    def convert_id_to_actions(self, framewise_predictions, gt_org, actions_dict_inv):
        recog = []
        for i in range(framewise_predictions.shape[1]):
            recog.extend([actions_dict_inv[framewise_predictions[:, i].item()]] * self.args.sample_rate)

        # adjust length of recog if there is a size mismatch with the ground truth
        if gt_org.shape[1] != len(recog):
            if gt_org.shape[1] < len(recog):
                recog = recog[:gt_org.shape[1]]
            elif gt_org.shape[1] > len(recog):
                recog = recog + recog[::-1]
                recog = recog[:gt_org.shape[1]]
        return recog
    
    
    def convert_segments_to_labels(self, action, duration, num_frames):
        assert  action.shape[0] == 1
        labels = action[0, :] - 2
        duration = duration[0, :]
        duration = duration / duration.sum()
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


    def convert_labels_to_segments(self, labels):
        segments = self.convert_labels(labels)
        # we need to insert <sos> and <eos>
        segments.insert(0, (torch.tensor(-2, device=labels.device), -1, -1))
        segments.append((torch.tensor(-1, device=labels.device), segments[-1][-1], segments[-1][-1]))
        
        target_labels = torch.stack([one_seg[0] for one_seg in segments]).unsqueeze(0) + 2 # two is because we are adding our sos and eos
        
        target_durations_unnormalized = self.compute_offsets([one_seg[2] for one_seg in segments]).to(target_labels.device).unsqueeze(0)
        segments_dict = {'seg_gt': target_labels,
                        'seg_dur': target_durations_unnormalized,
                        'seg_dur_normalized': target_durations_unnormalized/target_durations_unnormalized.sum().item(),
                        }
        return segments_dict
        

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


    def start_end2center_width(self, start_end):
        return torch.stack([start_end.mean(dim=2), start_end[:, :, 1] - start_end[:, :, 0]], dim=2)


    def convert_segments(self, segments):
        labels = np.zeros(segments[-1][-1] + 1)
        for segment in segments:
            labels[segment[1]:segment[2] + 1] = segment[0]
        return labels    
