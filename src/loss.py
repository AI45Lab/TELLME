from contextlib import nullcontext
import gc
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F 
import torchvision.transforms as T
import wandb
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.nn.functional import cosine_similarity
import numpy as np


def TripletLoss(Z1,Z2,margin=1):
    anchor = Z1
    pos = Z2
    N, Z = Z1.shape 

    indices = torch.arange(N)
    shifted_indices = torch.cat((indices[1:], indices[:1]))
    neg = Z2[shifted_indices]
    loss = nn.TripletMarginLoss(margin=margin, p=2)(anchor, pos, neg)

    return loss


def ContrastiveLoss(Z1, Z2, margin=1):
    anchor = Z1
    pos = Z2
    N, Z = Z1.shape 

    indices = torch.arange(N)
    shifted_indices = torch.cat((indices[1:], indices[:1]))
    neg = Z2[shifted_indices]
    euclidean_distance_pos = F.pairwise_distance(anchor, pos, keepdim = True)
    euclidean_distance_neg = F.pairwise_distance(anchor, neg, keepdim = True)
    loss_contrastive = torch.mean(torch.pow(euclidean_distance_neg, 2) +
                        torch.pow(torch.clamp(margin - euclidean_distance_pos, min=0.0), 2))/2

    return loss_contrastive


def nt_xent_loss(z1, z2, temperature=0.1):
    """ NT-Xent loss """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N, Z = z1.shape 
    device = z1.device 
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
    l_pos = torch.diag(similarity_matrix, N)
    r_pos = torch.diag(similarity_matrix, -N)
    positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
    diag = torch.eye(2*N, dtype=torch.bool, device=device)
    diag[N:,:N] = diag[:N,N:] = diag[:N,:N]
    negatives = similarity_matrix[~diag].view(2*N, -1)
    logits = torch.cat([positives, negatives], dim=1)
    logits /= temperature
    labels = torch.zeros(2*N, device=device, dtype=torch.int64)
    loss = F.cross_entropy(logits, labels, reduction='sum')
    return loss / (2 * N)

def BarlowTwins(z1, z2):
    batch_size, dim = z1.shape 

    bn = nn.BatchNorm1d(dim, affine=False).to(z1.device)
    lambd = 0.0051
    c = bn(z1).T @ bn(z2)
    c.div_(batch_size)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def clip_loss(
    query,
    document,
    logit_scale,
):
    N, Z = query.shape 
    
    device = query.device
    labels = torch.arange(query.shape[0]).to(device)
    if query.dtype != document.dtype:
        document = document.to(query.dtype)

    query = F.normalize(query, dim=-1)
    document = F.normalize(document, dim=-1)
    group1_expanded = query.unsqueeze(1)
    group2_expanded = document.unsqueeze(0)
    sim = F.cosine_similarity(group1_expanded, group2_expanded, dim=2)
    similarity_query_document = logit_scale * sim

    labels = labels * (document.size(0) // query.size(0))
    loss = F.cross_entropy(similarity_query_document, labels)
    similarity = sim.detach()
    mean_same = (similarity.trace() / similarity.size(0))
    mean_diff = (similarity.mean(axis=(-2, -1)) - (mean_same / similarity.size(0)))

    return loss, mean_same.detach(), mean_diff.detach()

def clip_loss_bi(query, document, logit_scale):
    """ CLIP loss using NT-Xent style """
    query = F.normalize(query, dim=1)
    document = F.normalize(document, dim=1)
    N, Z = query.shape
    device = query.device

    similarity_matrix = F.cosine_similarity(query.unsqueeze(1), document.unsqueeze(0), dim=-1)

    l_pos = torch.diag(similarity_matrix, 0)
    positives = torch.cat([l_pos, l_pos]).view(2 * N, 1)

    diag = torch.eye(N, dtype=torch.bool, device=device)
    negatives = similarity_matrix[~diag].view(N, -1)
    negatives_r = similarity_matrix.T[~diag].view(N, -1)

    negatives = torch.cat([negatives, negatives_r], dim=0)

    logits = torch.cat([positives, negatives], dim=1)
    logits *= logit_scale

    labels = torch.zeros(2*N, device=device, dtype=torch.int64)

    loss = F.cross_entropy(logits, labels, reduction='sum')
    
    return loss / (2 * N)

@torch.no_grad()
def cal_sim_loss(
    query,
    document,
    logit_scale,
):
    N, Z = query.shape 
    
    device = query.device
    labels = torch.arange(query.shape[0]).to(device)
    if query.dtype != document.dtype:
        document = document.to(query.dtype)

    query = F.normalize(query, dim=-1)
    document = F.normalize(document, dim=-1)
    group1_expanded = query.unsqueeze(1)
    group2_expanded = document.unsqueeze(0)
    similarity = F.cosine_similarity(group1_expanded, group2_expanded, dim=2)

    mean_same = (similarity.trace() / similarity.size(0))
    mean_diff = (similarity.mean(axis=(-2, -1)) - (mean_same / similarity.size(0)))

    return mean_same.detach(), mean_diff.detach()

def compute_retain_loss(retain_embs, orig_retain_embs):
    # print(torch.norm(retain_embs - orig_retain_embs, dim=-1, p=2, dtype=torch.float))
    return torch.norm(retain_embs - orig_retain_embs, dim=-1, p=2, dtype=torch.float).nanmean() 
