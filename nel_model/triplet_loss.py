"""
    -----------------------------------
    Achieve triplet loss
"""
import torch
from torch import nn
import torch.nn.functional as F


def cos_sim_batch(input_1, input_2, eps=1e-5):
    """
        batch cos similarity
        ------------------------------------------
        Args:
            input_1: (batch_size, hidden_size)
            input_2: (batch_size, hidden_size)
        Returns:
    """
    # inner = lambda x, y: torch.matmul(x.unsqueeze(1), y.unsqueeze(-1)).squeeze()
    inner = lambda x, y: (x * y).sum(dim=-1)

    dot = inner(input_1, input_2)
    m_1 = inner(input_1, input_1) ** 0.5
    m_2 = inner(input_2, input_2) ** 0.5

    return dot / (m_1 * m_2 + eps)


class TripletMarginLoss(nn.Module):
    def __init__(self, margin=0.25, sim='cos', p=2):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, pos, neg):
        """
            call
            ------------------------------------------
            Args:
                anchor: (batch_size, hidden_size)
                pos: (batch_size, hidden_size)
                neg: (batch_size, hidden_size)
            Returns:
        """
        sim_p = cos_sim_batch(anchor, pos)
        sim_n = cos_sim_batch(anchor, neg)

        loss = sim_n - sim_p + self.margin
        hinge_loss = F.relu(loss)
        return hinge_loss.mean()



class NpairLoss(nn.Module):
    """
    N-Pair loss
    Sohn, Kihyuk. "Improved Deep Metric Learning with Multi-class N-pair Loss Objective," Advances in Neural Information
    Processing Systems. 2016.
    http://papers.nips.cc/paper/6199-improved-deep-metric-learning-with-multi-class-n-pair-loss-objective
    """

    def __init__(self, args, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchors, positives, negatives):
        # anchors = embeddings[n_pairs[:, 0]]    # (n, embedding_size)
        # positives = embeddings[n_pairs[:, 1]]  # (n, embedding_size)
        # negatives = embeddings[n_negatives]    # (n, n-1, embedding_size)
        losses = self.n_pair_loss(anchors, positives, negatives) \
            + self.l2_reg * self.l2_loss(anchors, positives)

        return losses

    @staticmethod
    def get_n_pairs(labels):
        """
        Get index of n-pairs and n-negatives
        :param labels: label vector of mini-batch
        :return: A tuple of n_pairs (n, 2)
                        and n_negatives (n, n-1)
        """
        labels = labels.cpu().data.numpy()
        n_pairs = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            anchor, positive = np.random.choice(label_indices, 2, replace=False)
            n_pairs.append([anchor, positive])

        n_pairs = np.array(n_pairs)

        n_negatives = []
        for i in range(len(n_pairs)):
            negative = np.concatenate([n_pairs[:i, 1], n_pairs[i+1:, 1]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_pairs), torch.LongTensor(n_negatives)

    @staticmethod
    def n_pair_loss(anchors, positives, negatives):
        """
        Calculates N-Pair loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :param negatives: A torch.Tensor, (n, n-1, embedding_size)
        :return: A scalar
        """
        anchors = torch.unsqueeze(anchors, dim=1)  # (n, 1, embedding_size)
        positives = torch.unsqueeze(positives, dim=1)  # (n, 1, embedding_size)

        x = torch.matmul(anchors, (negatives - positives).transpose(1, 2))  # (n, 1, n-1)
        x = torch.sum(torch.exp(x), 2)  # (n, 1)
        loss = torch.mean(torch.log(1+x))
        return loss

    @staticmethod
    def l2_loss(anchors, positives):
        """
        Calculates L2 norm regularization loss
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (n, embedding_size)
        :return: A scalar
        """
        return torch.sum(anchors ** 2 + positives ** 2) / anchors.shape[0]


if __name__ == '__main__':
    x = torch.randn(64, 512)
    y = torch.randn(64, 512)
    print(cos_sim_batch(x, y).size())