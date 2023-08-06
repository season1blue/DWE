"""
    top-k metric
"""
from circle_loss import cosine_similarity, dot_similarity
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch


def faiss_cal_topk(args, query, answer_list, entity_features):
    answer_list = answer_list.cpu()  # answer_entity在entity_list中的index
    query = query.cpu()
    index = faiss.IndexFlatL2(args.hidden_size)
    index.add(entity_features)
    k = 50
    _, search_res = index.search(query, k)  # query与所有的entity做相似，按照相似度降序排列的entity_id
    rank_list = []
    for index, answer in enumerate(answer_list):
        rank = torch.nonzero(search_res[index] == answer).squeeze(-1)  # answer_entity是否在topk个相似列表里
        rank = rank.item() if rank.size(0) else k  # k代表不在
        rank_list.append(rank)

    return rank_list

def cal_top_k(args, query, pos_feats, search_feats):
    """
        Input query, positive sample features, negative sample features
        query: 32,512
        pos_feats: 32, 1, 512
        search_feats: 32, 50(100), 512
        return the ranking of positive samples
        ------------------------------------------
        Args:
        Returns:
    """

    if args.similarity == 'cos':
        ans = similarity_rank(query, pos_feats, search_feats, cosine_similarity)
    elif args.similarity == 'dot':
        ans = similarity_rank(query, pos_feats, search_feats, dot_similarity)
    else:
        ans = lp_rank(query, pos_feats, search_feats, args.loss_p)

    return ans


def similarity_rank(query, pos_feats, search_feats, cal_sim):
    """
        Sample ranking based on similarity
        ------------------------------------------
        Args:
        Returns:
    """
    rank_list = []
    # print(query.size())
    # print(search_feats.size())
    sim_s = cal_sim(query, search_feats).detach().cpu().numpy()  # batch_size, n_search
    sim_p = cal_sim(query, pos_feats).detach().cpu().numpy()  # batch_size, 1

    sim_mat = sim_s - sim_p # expect sim_s < sim_p
    ranks = (sim_mat > 0).sum(-1) + 1

    return ranks, sim_p, sim_s


def lp_distance(x, dim, p):
    return (x ** p).sum(dim=dim) ** (1 / p)


def lp_rank(query, pos_feats, search_feats, p=2):
    """
        Using LP distance to calculate the rank of positive examples
        ------------------------------------------
        Args:
        Returns:
    """
    rank_list = []
    dis_p = lp_distance(query - pos_feats.squeeze(), dim=-1, p=p).detach().cpu().numpy()
    dis_sf = lp_distance(query.unsqueeze(1) - search_feats, dim=-1, p=p).detach().cpu().numpy()

    batch_size = dis_p.size(0)
    for i in range(batch_size):
        rank = 0
        for dis in dis_sf[i]:
            if dis < dis_p[i]:
                rank += 1
        rank_list.append(rank)

    return rank_list, dis_p, dis_sf