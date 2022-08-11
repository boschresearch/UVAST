#######################################
# code written by Nadine Behrmann
#######################################
import torch
from torch import optim
from torch.autograd import Variable
from utils import convert_labels_to_segments, convert_segments_to_labels
from eval import accuracy
import os


def get_p_matrix(framewise_pred, transcript):
    # returns matrix with framewise probabilities of size num_segments x seq_len
    framewise_pred = torch.softmax(framewise_pred, dim=1)
    P = []
    for i in range(transcript.shape[1]):
        P.append(framewise_pred[0, transcript[0, i], :])
    # add eps for stability
    P = - torch.log(torch.stack(P) + (1e-16))
    return P

def get_m_matrix(lengths, seq_len, sharpness, device):
    bn = torch.cumsum(lengths, dim=0)
    center = bn - lengths / 2
    width = lengths / 2
    t = torch.linspace(0, 1, seq_len).to(device)
    term1 = torch.exp(sharpness * (t[None, :] - center[:, None] - width[:, None])) + 1
    term2 = torch.exp(sharpness * (-t[None, :] + center[:, None] - width[:, None])) + 1
    M = 1 / (term1 * term2)
    return M

def fifa(action, duration=None, framewise_pred=None, mean_path=None, gt=None, uniform_dur=False, num_epochs=10000, device=None, sharpness=None, lr=None, args=None):
    # get fixed probability matrix P[n, t] = probability that frame t belongs to segment n
    P = get_p_matrix(framewise_pred, action - 2)
    seq_len = framewise_pred.shape[2]

    if args:
        mean_path = args.data_root_mean_duration + '/' + args.dataset + '/splits/fifa_mean_dur_split' + str(args.split) + '.pt'
        if args.dataset == '50salads':
            sharpness = 80; lr = 0.01; num_epochs = 3000

        if args.dataset == 'gtea':
            sharpness = 80; lr = 0.1; num_epochs = 3000

        if args.dataset == 'breakfast':
            sharpness = 80; lr = 0.01; num_epochs = 3000

    mean_dur = torch.load(mean_path)
    means = torch.stack([mean_dur[a.item()] for a in action[0, :]]).to(device)

    if uniform_dur:
        duration = torch.ones_like(duration)
    duration = duration / duration.sum()

    log_length = torch.log(duration[0, :] + (1e-16))
    log_length = Variable(log_length, requires_grad=True)

    with torch.enable_grad():
        for epoch in range(num_epochs):
            length = torch.exp(log_length)
            M = get_m_matrix(length, seq_len, sharpness,device) 
            E_o = (P * M).mean()
            E_l = (length - means).abs().sum()
            if args.dataset=='breakfast':
                loss = E_o + 0.1 * E_l + torch.abs(length.sum() - 1)
            if args.dataset=='50salads':
                loss = E_o + 0.01 * E_l + torch.abs(length.sum() - 1)
            if args.dataset=='gtea':
                loss = E_o + 0.1 * torch.abs(length.sum() - 1)
                
            loss.backward()
            log_length.data -= lr * log_length.grad.data
            log_length.grad.data.zero_()
            if gt is not None:
                pred = convert_segments_to_labels(action, torch.exp(log_length).unsqueeze(0).detach(), seq_len)
                acc = accuracy(gt[0, :], pred[0, :])
                print(acc)
    return torch.exp(log_length.data).unsqueeze(0)

