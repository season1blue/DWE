import torch
import torch.nn as nn
import numpy as np
# from torch.nn import TripletMarginLoss
# from word_level import WordLevel
# from phrase_level import PhraseLevel
# from sent_level import SentLevel
# from gated_fuse import GatedFusion
# from recursive_encoder import RecursiveEncoder
from circle_loss import CircleLoss
from triplet_loss import TripletMarginLoss, NpairLoss


def Contrastive_loss(out_1, out_2, batch_size, temperature=0.5):
    out = torch.cat([out_1, out_2], dim=0)  # [2*B, D]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)  # [2*B, 2*B]
    '''
    torch.mm是矩阵乘法，a*b是对应位置上的数相除，维度和a，b一样
    '''
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    '''
    torch.eye生成对角线上为1，其他为0的矩阵
    torch.eye(3)
    tensor([[ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]])
    '''
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / (sim_matrix.sum(dim=-1) - pos_sim))).mean()
    return loss



class NELModel(nn.Module):
    def __init__(self, args):
        super(NELModel, self).__init__()
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.output_size = args.output_size
        self.seq_len = args.max_sent_length
        self.text_feat_size = args.text_feat_size
        self.img_feat_size = args.img_feat_size
        self.feat_cate = args.feat_cate.lower()

        self.split_trans = nn.Sequential(
            nn.Linear(self.img_feat_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        self.img_trans = nn.Sequential(
            nn.Linear(self.img_feat_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )
        self.text_trans = nn.Sequential(
            nn.Linear(self.text_feat_size, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(self.dropout)
        )

        self.entity_trans = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.text_feat_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        # Dimension reduction
        self.pedia_out_trans = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 3, self.output_size),
        )
        self.img_att = nn.MultiheadAttention(self.hidden_size, args.nheaders, batch_first=True)

        # circle loss
        self.loss_function = args.loss_function
        self.loss_margin = args.loss_margin
        self.sim = args.similarity
        if self.loss_function == 'circle':
            self.loss_scale = args.loss_scale
            self.loss = CircleLoss(self.loss_scale, self.loss_margin, self.sim)
        else:
            # self.loss_p = args.loss_p
            # self.loss = TripletMarginLoss(margin=self.loss_margin, p=self.loss_p)
            self.loss = NpairLoss(args)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, model_type, mention=None, text=None, total=None, detection=None, pos_feats=None, neg_feats=None):
        """
            ------------------------------------------
            Args:
                text: tensor: (batch_size, max_seq_len, text_feat_size), the output of bert hidden size
                img: float tensor: (batch_size, ..., img_feat_size), image features - resnet
                bert_mask: tensor: (batch_size, max_seq_len)
                pos_feats(optional): (batch_size, n_pos, output_size)
                neg_feats(optional): (batch_size, n_neg, output_size)
            Returns:
        """

        # pos_feats = self.entity_trans(pos_feats)
        # neg_feats = self.entity_trans(neg_feats)

        mention_trans = self.text_trans(mention)
        bert_trans = self.text_trans(text)
        total_trans = self.img_trans(total)

        query = torch.cat([bert_trans, total_trans, mention_trans], dim=-1)
        query = self.pedia_out_trans(query)
        query = query.squeeze(1)

        # 注意这里的维度，如果不满足TripletMarginLoss的维度设置，会存在broadcast现象，导致性能大幅下降 全都要是 [bsz*hs]
        triplet_loss = self.loss(query, pos_feats.squeeze(1), neg_feats.squeeze(1))

        return triplet_loss, query

    def trans(self, x):
        return x
        # return self.text_trans(x)
