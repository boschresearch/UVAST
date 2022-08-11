# code from: https://github.com/ChinaYi/ASFormer (MIT License) - some modifications
#######################################
# code written by S. Alireza Golestaneh
#######################################
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_


class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, compression, dilaion):
        super(AttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // compression, kernel_size=1)
        self.key_conv   = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // compression, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // compression, kernel_size=1)
        self.conv_out   = nn.Conv1d(in_channels=v_dim // compression, out_channels=v_dim, kernel_size=1)
        self.dilaion = dilaion
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, mask):
        query = self.query_conv(input.clone())
        key = self.key_conv(input.clone())
        value = self.value_conv(input)
        return self.sliding_window_self_att(query, key, value, mask)

    def sliding_window_self_att(self, q,k,v, mask):
        QB,QE,QS = q.size()
        KB,KE,KS = k.size()
        VB,VE,VS  = v.size()

        # padding zeros for the last segment
        # we want our sequence be dividable by  self.dilaion, so we need QS % self.dilaion == 0, if it is not the case we will pad it so it become
        nb = QS // self.dilaion 
        if QS % self.dilaion != 0:
            q = F.pad(q,pad=(0,self.dilaion - QS % self.dilaion),mode='constant',value=0)
            k = F.pad(k,pad=(0,self.dilaion - QS % self.dilaion),mode='constant',value=0)
            v = F.pad(v,pad=(0,self.dilaion - QS % self.dilaion),mode='constant',value=0)
            nb += 1
            
        padding_mask = torch.cat([torch.ones((QB, 1, QS)).to(q.device) * mask[:,0:1,:], torch.zeros((QB, 1, self.dilaion * nb - QS)).to(q.device)],dim=-1)
        
        # sliding window approach, by splitting query_proj and key_proj into shape (QE, l) x (QE, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(QB, QE, nb, self.dilaion).permute(0, 2, 1, 3).reshape(QB, nb, QE, self.dilaion)

        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = F.pad(k,pad=(self.dilaion // 2,self.dilaion // 2),mode='constant',value=0)
        v = F.pad(v,pad=(self.dilaion // 2,self.dilaion // 2),mode='constant',value=0)
        padding_mask = F.pad(padding_mask,pad=(self.dilaion // 2,self.dilaion // 2),mode='constant',value=0)
        
        # 2. reshape key_proj of shape (QB*nb, QE, 2*self.dilaion)
        k = torch.cat([k[:,:, i*self.dilaion:(i+1)*self.dilaion+(self.dilaion//2)*2].unsqueeze(1) for i in range(nb)], dim=1) # special case when self.dilaion = 1
        v = torch.cat([v[:,:, i*self.dilaion:(i+1)*self.dilaion+(self.dilaion//2)*2].unsqueeze(1) for i in range(nb)], dim=1) 
        
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat([padding_mask[:,:, i*self.dilaion:(i+1)*self.dilaion+(self.dilaion//2)*2].unsqueeze(1) for i in range(nb)], dim=1)
        
        # construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        window_mask = torch.zeros((1, self.dilaion, self.dilaion + 2* (self.dilaion //2))).to(q.device)
        for i in range(self.dilaion):
            window_mask[:, :, i:i+self.dilaion] = 1

        final_mask = window_mask.unsqueeze(1).repeat(QB , nb, 1, 1) * padding_mask 
        
        proj_query=q
        proj_key=k
        proj_val=v
        padding_mask = final_mask

        b,m, QE, l1 = proj_query.shape
        b,m, KE, l2 = proj_key.shape
        
        energy = torch.einsum('n b k i, n b k j -> n b i j', proj_query, proj_key)
        attention = energy / (np.sqrt(QE)*1.0)
        attention = attention + torch.log(padding_mask + 1e-6) # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention) 
        attention = attention * padding_mask
        output = torch.einsum('n b i k, n b j k-> n b i j', proj_val,attention)

        bb,cc,ww, hh = output.shape
        output = einops.rearrange(output, 'b c h w -> (b c) h w')
        output = self.conv_out(F.gelu(output))
        output = einops.rearrange(output, '(b c) h w->b c h w',b=bb,c=cc)

        output = output.reshape(QB, nb, -1, self.dilaion).permute(0, 2, 1, 3).reshape(QB, -1, nb * self.dilaion)
        output = output[:, :, 0:QS]
        return output * mask[:, 0:1, :]

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential( nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
                                    nn.GELU()
                                    )
    def _reset_parameters(self):
        constant_(self.layer[0].bias, 0.)

    def forward(self, x):
        return self.layer(x)

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, compression, args=None):
        super(AttModule, self).__init__()
        self.feed_forward  = ConvFeedForward(dilation, in_channels, out_channels)

        if args.enc_norm_type == 'InstanceNorm1d':
            self.instance_norm = nn.InstanceNorm1d(in_channels,track_running_stats=False,affine=False)
            self.instance_norm2 = nn.InstanceNorm1d(64,track_running_stats=False,affine=False)

        elif args.enc_norm_type == 'InstanceNorm1d_track':
            self.instance_norm = nn.InstanceNorm1d(in_channels,track_running_stats=True)
            self.instance_norm2 = nn.InstanceNorm1d(64,track_running_stats=True)

        elif args.enc_norm_type == 'LayerNorm':
            self.instance_norm = nn.LayerNorm(in_channels)
            self.instance_norm2 = nn.LayerNorm(in_channels)

        self.att_layer     = AttLayer(in_channels, in_channels, out_channels, compression, dilation) # dilation
        self.conv_1x1      = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout       = nn.Dropout()
        
        self.feed_forward2  = ConvFeedForward(dilation, 64, 64)

        self.args = args
        

    def _reset_parameters(self):
        constant_(self.conv_1x1.bias, 0.)
        
    def forward(self, x, mask, istraining=False):

        out                 = self.feed_forward(x)

        if self.args.enc_norm_type == 'InstanceNorm1d' or self.args.enc_norm_type == 'InstanceNorm1d_track':
            out             = self.att_layer(self.instance_norm(out), mask) + out
        else:
            out             = self.att_layer(self.instance_norm(out.permute(0,2,1)).permute(0,2,1), mask) + out
            
        out                 = self.conv_1x1(out)
        if istraining:
            out                 = self.dropout(out)
        
        out = self.feed_forward2(out)
            
        out                 = x + out
            
        if self.args.enc_norm_type == 'InstanceNorm1d' or self.args.enc_norm_type == 'InstanceNorm1d_track':
            out                 = self.instance_norm2(out)
        else:
            out                 = self.instance_norm2(out.permute(0,2,1)).permute(0,2,1)
        
        return  out* mask[:, 0:1, :]


class Encoder_advanced(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, args=None):
        super(Encoder_advanced, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) 
        
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1,args=args) for i in range(num_layers)])
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate
        self.args = args



    def forward(self, x, mask, istraining=False):
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            if istraining:
                x = self.dropout(x)
            x = x.squeeze(2)
            
        B = x.shape[0]
        
        feature = self.conv_1x1(x) 
        
        for index, layer in enumerate(self.layers):
            feature = layer(feature, mask, istraining=istraining)        

        out = self.conv_out(feature) * mask[:, 0:1, :]
        
        return out, feature
