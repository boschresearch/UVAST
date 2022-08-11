#!/usr/bin/python2.7
#######################################
# code written by Nadine Behrmann
#######################################
import torch
import torch.nn as nn




class SegmentEmbedding(nn.Module):
    """
    SegmentEmbedding:
    - implements different options how the action class and duration of a segment can be embedded
    - embeddings will be fed to the decoder
    """
    def __init__(self, args):
        super().__init__()
        token_dim = args.num_f_maps
        self.action_emb = nn.Embedding(args.num_classes + 2, args.num_f_maps)
        self.args = args

    def forward(self, output):
        output_labels, output_duration = output[0], output[1]
        b, le = output_labels.shape
        mask_labels = torch.ones_like(output_labels)
        mask_labels[output_labels == -1] = 0 # -1 is the ignore label, so we need to mask is out
        tgt_emb = self.action_emb((output_labels * mask_labels).long()) * mask_labels.unsqueeze(-1)
        return tgt_emb , mask_labels
        
