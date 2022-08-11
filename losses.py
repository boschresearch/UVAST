# code from https://github.com/yabufarha/ms-tcn (MIT+CC License) - many modifications and additions
import torch
import torch.nn as nn
from utils import AverageMeter
import random


class FrameWiseLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args        

        self.CELogit = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.0)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.Softmax = nn.Softmax(dim=1)
        self.NLLLoss = nn.NLLLoss(ignore_index=-1)
        # initialize average meters
        self.reset()

    def __call__(self, predictions, batch_target, mask, epoch):      
        # initialize losses
        loss = torch.tensor(0.0, device=mask.device)
        framewise_loss = torch.tensor(0.0, device=mask.device)
        framewise_loss_g = torch.tensor(0.0, device=mask.device)
        framewise_loss_random_singleframe = torch.tensor(0.0, device=mask.device)

        # framewise cross-entropy loss on logits of cncoder
        if self.args.do_framewise_loss:
            for i in range(len(predictions)):
                framewise_loss = framewise_loss + self.CELogit(predictions[i], batch_target.long())
                
        # group framewise loss CE - We first average the features (different methods can be used for normalizing features), then appy CE or NLLLoss on them
        g_pred = None
        g_gt = None
        if self.args.do_framewise_loss_g: 
            for i in range(len(predictions)):
                predictions_b = predictions[i]
                group_pred_feat = []
                group_gt = []
                for fcls in torch.unique(batch_target).view(-1): 
                    if fcls != -1:
                        #  here we normalize the features somehow, either via norm, or signmoid
                        if self.args.framewise_loss_g_apply_logsoftmax:
                            predictions_group = self.LogSoftmax(predictions_b)
                        elif self.args.framewise_loss_g_apply_nothing:
                            predictions_group = predictions_b
                        else:
                            assert (self.args.framewise_loss_g_apply_nothing or self.args.framewise_loss_g_apply_logsoftmax), ' you need to define the framewise group model'
                        # computing the average
                        suming_similar_act = (predictions_group*((batch_target == fcls) + 0.0).unsqueeze(1)).sum(2).sum(0, keepdim=True)
                        numm_similar_act = ((batch_target == fcls)+0.0).sum()
                        mean_fcls_feat = suming_similar_act / numm_similar_act
                        # append to list
                        group_pred_feat.append(mean_fcls_feat)
                        group_gt.append(fcls.view(-1).long())
                # concat so we can use reducing mean of the loss function
                g_pred = torch.cat(group_pred_feat, dim=0)
                g_gt = torch.cat(group_gt, dim=0)
                # computing the loss
                if self.args.framewise_loss_g_apply_logsoftmax:
                    framewise_loss_g = framewise_loss_g + self.NLLLoss(g_pred, g_gt.long())
                elif self.args.framewise_loss_g_apply_nothing:
                    framewise_loss_g = framewise_loss_g + self.CELogit(g_pred, g_gt.long())
            framewise_loss_g = framewise_loss_g / len(predictions)
                
        # final loss
        loss = framewise_loss + framewise_loss_g

        self.update_metrics(predictions[0], batch_target, g_pred, g_gt, mask, loss, framewise_loss, framewise_loss_g, framewise_loss_random_singleframe)
        return loss
    
    def update_metrics(self, predictions, batch_target, g_pred, g_gt, mask, loss,framewise_loss, framewise_loss_g, framewise_loss_random_singleframe):
        self.loss.update(loss.item())
        self.framewise_loss.update(framewise_loss.item())
        self.framewise_loss_g.update(framewise_loss_g.item())
        self.framewise_loss_random_singleframe.update(framewise_loss_random_singleframe.item())

        _, predicted = torch.max(predictions.data, 1)
        correct = ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
        total = torch.sum(mask[:, 0, :]).item()
        self.acc.update(correct / total, n=total)

        if g_pred is not None:
            _, predicted_g = torch.max(g_pred.data, 1)
            correct_g = ((predicted_g == g_gt).float() ).sum().item()
            total_g = g_pred.shape[0]
            self.acc_g.update(correct_g / total_g, n=total_g)

    def log_metrics(self, mode, epoch, writer=None):
        print(f"[{mode} epoch {epoch}] Loss: {self.loss.avg:.4f}, "
              f"Loss Frame-W : {self.framewise_loss.avg:.4f}, "
              f"Loss Frame-W group : {self.framewise_loss_g.avg:.4f}, "
              f"Acc Frame-W: {self.acc.avg:.4f}, "
              f"Acc Group Frame-W: {self.acc_g.avg:.4f}", flush=True)

    def reset(self):
        self.loss = AverageMeter()
        self.framewise_loss = AverageMeter()
        self.framewise_loss_g = AverageMeter()
        self.framewise_loss_random_singleframe = AverageMeter()
        self.acc = AverageMeter()
        self.acc_g = AverageMeter()
        

class SegmentLossAction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.CELogit = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.0)
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        self.Softmax = nn.Softmax(dim=1)
        self.NLLLoss = nn.NLLLoss(ignore_index=-1)
        self.args = args      
        # initialize average meters
        self.reset()

    def __call__(self, pred_action, target_action, vid_len, epoch):
        loss = torch.tensor(0.0, device=target_action.device)
        segwise_loss = torch.tensor(0.0, device=target_action.device)
        segwise_loss_g = torch.tensor(0.0, device=target_action.device)
        segwise_loss_g_indecs = torch.tensor(0.0, device=target_action.device)

        # loss on the output of decoder, which is the segment-wise loss
        if self.args.do_segwise_loss:
            for i in range(len(pred_action)):
                segwise_loss = segwise_loss + self.CELogit(pred_action[i], target_action.long())
            segwise_loss = segwise_loss / (float(len(pred_action))) 
            
        # segment wise loss on groups (for each unique class averge among the groups)
        g_pred = None
        g_gt = None
        if self.args.do_segwise_loss_g:
            group_pred = []
            group_gt = []
            for fcls in torch.unique(target_action).view(-1): 
                if fcls != -1:
                    # apply the loss on the features from every layer of the decoder
                    # if self.args.do_segwise_loss_onall_element:             
                    for i in range(len(pred_action)): # this can be also in a outside loop
                        #  here we normalize the features somehow, either via norm, or signmoid
                        if self.args.segwise_loss_g_apply_logsoftmax:
                            predictions_group = self.LogSoftmax(pred_action[i])
                        elif self.args.segwise_loss_g_apply_nothing:
                            predictions_group = pred_action[i]
                        # computing the average
                        suming_similar_act = (predictions_group * ((target_action == fcls) + 0.0).unsqueeze(1)).sum(2).sum(0, keepdim=True)
                        numm_similar_act = ((target_action == fcls) + 0.0).sum()
                        mean_fcls_feat = suming_similar_act / numm_similar_act
                        # append to list
                        group_pred.append(mean_fcls_feat)
                        group_gt.append(fcls.view(-1).long())
            # concat so we can use reducing mean of the loss function
            g_pred = torch.cat(group_pred, dim=0)
            g_gt = torch.cat(group_gt, dim=0)
            # computing the loss
            if self.args.segwise_loss_g_apply_logsoftmax:
                segwise_loss_g = self.NLLLoss(g_pred, g_gt.long())
            elif self.args.segwise_loss_g_apply_nothing:
                segwise_loss_g = self.CELogit(g_pred, g_gt.long())

        # total loss  computing
        loss = segwise_loss + segwise_loss_g

        self.update_metrics(pred_action[-1], target_action, g_pred, g_gt, loss, segwise_loss, segwise_loss_g, segwise_loss_g_indecs)
        return loss

    def update_metrics(self, pred_action, target_action, g_pred, g_gt, loss, segwise_loss, segwise_loss_g, segwise_loss_g_indecs):
        # update losses
        self.segwise_loss.update(segwise_loss.item())
        self.segwise_loss_g.update(segwise_loss_g.item())
        self.segwise_loss_g_indecs.update(segwise_loss_g_indecs.item())
        
        self.losses.update(loss.item())
        # update action acc
        _, predicted = torch.max(pred_action.data, 1)
        mask = torch.ones_like(target_action)
        mask[target_action == -1] = 0
        correct = ((predicted == target_action).float() * mask).sum().item()
        total = torch.sum(mask).item()
        self.acc_action.update(correct / total, n=total)
        
        if g_pred is not None:
            _, predicted_g = torch.max(g_pred.data, 1)
            correct_g = ((predicted_g == g_gt).float()).sum().item()
            total_g = g_pred.shape[0]
            self.acc_g.update(correct_g / total_g, n=total_g)

    def log_metrics(self, mode, epoch, writer=None):
        print(f"[{mode} epoch {epoch}] Loss: {self.losses.avg:.4f}, "
              f"Loss Segment-W : {self.segwise_loss.avg:.4f}, "
              f"Loss Segment-W  group: {self.segwise_loss_g.avg:.4f}, "
            #   f"Loss Segment-Index  group: {self.segwise_loss_g_indecs.avg:.4f}, "
              f"Acc Segment-W: {self.acc_action.avg:.4f}, "
              f"Acc Group Segment-W: {self.acc_g.avg:.4f}", flush=True)
    
    def reset(self):
        self.segwise_loss = AverageMeter()
        self.segwise_loss_g = AverageMeter()
        self.losses = AverageMeter()
        self.acc_action = AverageMeter()
        self.segwise_loss_g_indecs = AverageMeter()
        self.acc_g = AverageMeter()


class AttentionLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.NLLLoss = nn.NLLLoss(ignore_index=-1)
        self.CELogit = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.0)
        self.args = args
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        
        # initialize average meters
        self.reset()
    
    def __call__(self, pred_attn, target_attn, batch_target):    
        attn_loss_nll = torch.tensor(0.0, device=target_attn.device)
        attn_loss_ce = torch.tensor(0.0, device=target_attn.device)
        attn_loss_tmp = torch.tensor(0.0, device=target_attn.device)
        attn_loss_tmp2 = torch.tensor(0.0, device=target_attn.device)
        loss_max_prob = torch.tensor(0.0, device=target_attn.device)
        
        valid_pred_attn_lln = None
        valid_target_attn_lln = None
        
        if self.args.do_crossattention_action_loss_nll:
            for i in range(len(pred_attn)):
                for batchnum in range(pred_attn[i].shape[0]):
                    valid_pred_attn_lln = pred_attn[i].permute(0, 2, 1)[batchnum].unsqueeze(0)[batch_target[batchnum].repeat(self.args.n_head_dec, 1) != -1]
                    valid_target_attn_lln = (target_attn.permute(0, 2, 1)[batchnum][batch_target[batchnum] != -1]).repeat(self.args.n_head_dec, 1).to(self.args.device)
                    valid_pred_attn_log = self.LogSoftmax(valid_pred_attn_lln/self.args.temperature)
                    attn_loss_tmp = attn_loss_tmp + (self.NLLLoss(valid_pred_attn_log, valid_target_attn_lln.argmax(1).view(-1).long()) * float(1 / pred_attn[i].shape[0]))
            attn_loss_nll = attn_loss_tmp / (len(pred_attn) + 0.0)

        loss = attn_loss_nll
            
        valid_pred_attn_lln_all, valid_target_attn_lln_all = None, None
        valid_pred_attn_lln_all = pred_attn[-1].permute(0, 2, 1)[batch_target.repeat(self.args.n_head_dec, 1) != -1]
        valid_target_attn_lln_all = (target_attn.permute(0, 2, 1)[batch_target != -1]).repeat(self.args.n_head_dec, 1)

        self.update_metrics(loss, attn_loss_nll, attn_loss_ce, pred_attn, target_attn, valid_pred_attn_lln_all, valid_target_attn_lln_all)
        return loss

    def update_metrics(self, loss, attn_loss_nll, attn_loss_ce, pred_attn, target_attn, valid_pred_attn_lln, valid_target_attn_lln):
        # update losses
        self.losses.update(loss.item())
        self.attn_loss_nll.update(attn_loss_nll.item())
        self.attn_loss_ce.update(attn_loss_ce.item())
                
        # update attn acc lln
        if valid_target_attn_lln is not None:
            _, predicted = torch.max(valid_pred_attn_lln.data, 1)
            correct2 = (predicted == valid_target_attn_lln.argmax(1).view(-1).long().to(self.args.device)).sum().item()
            total2 = valid_target_attn_lln.argmax(1).view(-1).long().shape[0]
            self.acc_attn_lln.update(correct2 / total2, n=total2)
            
    def log_metrics(self, mode, epoch,  writer=None):
        print(f"[{mode} epoch {epoch}] Loss: {self.losses.avg:.4f}, "
              f"Loss Attention LLN: {self.attn_loss_nll.avg:.4f}, " 
            #   f"Loss Attention CE: {self.attn_loss_ce.avg:.4f}, " 
              f"Acc Attention Map: {self.acc_attn_lln.avg:.4f}", flush=True)

    def reset(self):
        self.losses = AverageMeter()
        self.attn_loss_nll = AverageMeter()
        self.attn_loss_ce = AverageMeter()
        self.acc_attn_lln = AverageMeter()
        

class DurAttnCALoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.CE = nn.CrossEntropyLoss(ignore_index=-1)
        self.args = args
        # initialize average meters
        self.reset()

    def __call__(self, pred_attn_bmm, target_attn):
        attn_loss_segmentid_ce = torch.tensor(0.0, device=target_attn.device)
        attn_loss_segmentid_ce_inside_CA = torch.tensor(0.0, device=target_attn.device)
        loss = torch.tensor(0.0, device=target_attn.device)
       
        if self.args.do_crossattention_dur_loss_ce:
            GT_label = target_attn.argmax(1)[0]
            attn_loss_segmentid_ce = self.CE(pred_attn_bmm[0], GT_label)
            
        loss = attn_loss_segmentid_ce 
            
        self.update_metrics(loss, attn_loss_segmentid_ce, attn_loss_segmentid_ce_inside_CA, pred_attn_bmm, target_attn)
        return loss

    def update_metrics(self, loss, attn_loss_segmentid_ce, attn_loss_segmentid_ce_inside_CA, pred_attn_bmm, target_attn):
        # update losses
        self.loss.update(loss.item())
        self.attn_loss_segmentid_ce.update(attn_loss_segmentid_ce.item())
        self.attn_loss_segmentid_ce_inside_CA.update(attn_loss_segmentid_ce_inside_CA.item())

        _, predicted = torch.max(pred_attn_bmm.data, 2)
        correct2 = (predicted.view(-1) == target_attn.argmax(1).view(-1).long()).sum().item()
        total2 = target_attn.argmax(1).view(-1).long().shape[0]
        self.acc_ca_pred.update(correct2 / total2, n=total2)

    def log_metrics(self, mode, epoch,  writer=None):
        print(f"[{mode} epoch {epoch}] Loss: {self.loss.avg:.4f}, "
              f"Loss Attention CE: {self.attn_loss_segmentid_ce.avg:.4f}, " 
              f"Acc CA: {self.acc_ca_pred.avg:.4f}, " 
              f"Loss Attention CE inside dec: {self.attn_loss_segmentid_ce_inside_CA.avg:.4f}", flush=True)

    def reset(self):
        self.loss = AverageMeter()
        self.attn_loss_segmentid_ce_inside_CA = AverageMeter()
        self.attn_loss_segmentid_ce = AverageMeter()
        self.acc_ca_pred = AverageMeter()
        
        
