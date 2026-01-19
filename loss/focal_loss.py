#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp


# Focal Loss 实现
class FocalLoss(nn.Module):
    '''
    多分类 Focal Loss
    参考: https://arxiv.org/abs/1708.02002

    Args:
        alpha (float or list): 类别不平衡权重因子。float 表示所有类别相同；list 表示每类权重。
        gamma (float): 聚焦参数，强化难样本。
        reduction (str): 输出规约方式: 'none' | 'mean' | 'sum'。
        ignore_index (int): 忽略的标签值，不参与损失计算。
    '''
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, logits, label):
        '''
        用法:
            >>> criteria = FocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)  # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384))  # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        logits = logits.float()  # 使用 float32 避免数值不稳定

        # one-hot 编码并计算 softmax 概率
        num_classes = logits.size(1)
        log_prob = F.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)

        # 处理 ignore_index
        mask = label != self.ignore_index
        label = label.clone()
        label[~mask] = 0

        # 标签转为 one-hot
        one_hot = torch.zeros_like(log_prob)
        one_hot.scatter_(1, label.unsqueeze(1), 1.0)

        # alpha 权重
        if isinstance(self.alpha, (list, tuple)):
            alpha = torch.tensor(self.alpha, device=logits.device).view(1, num_classes, 1, 1)
        else:
            alpha = self.alpha

        # 计算 focal loss
        loss = -alpha * (1 - prob) ** self.gamma * one_hot * log_prob
        loss = loss.sum(dim=1)

        # 忽略指定标签
        loss = loss * mask.float()

        # 规约输出
        if self.reduction == 'mean':
            loss = loss.sum() / mask.float().sum() if mask.float().sum() > 0 else torch.tensor(0.0, device=logits.device)
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


# 二分类 Focal Loss
class BinaryFocalLoss(nn.Module):
    '''
    二分类 Focal Loss
    参考: https://arxiv.org/abs/1708.02002

    Args:
        alpha (float): 类别不平衡权重因子 (默认: 0.25)。
        gamma (float): 聚焦参数 (默认: 2.0)。
        reduction (str): 输出规约方式: 'none' | 'mean' | 'sum'。
    '''
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        '''
        用法:
            >>> criteria = BinaryFocalLoss()
            >>> logits = torch.randn(8, 1, 384, 384)  # nchw, float/half
            >>> targets = torch.randint(0, 2, (8, 1, 384, 384)).float()  # nchw, float32
            >>> loss = criteria(logits, targets)
        '''
        logits = logits.float()
        targets = targets.float()

        # Sigmoid 激活
        prob = torch.sigmoid(logits)

        # 计算 focal loss 组件
        pt = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)

        # 规约输出
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


def main():
    torch.manual_seed(0)

    # 多分类测试
    criteria = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean', ignore_index=255)
    logits = torch.randn(2, 3, 4, 4)
    labels = torch.randint(0, 3, (2, 4, 4))
    labels[0, 0, 0] = 255
    loss = criteria(logits, labels)
    print(f"FocalLoss: {loss.item():.6f}")

    # 二分类测试
    bcriteria = BinaryFocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    blogits = torch.randn(2, 1, 4, 4)
    targets = torch.randint(0, 2, (2, 1, 4, 4)).float()
    bloss = bcriteria(blogits, targets)
    print(f"BinaryFocalLoss: {bloss.item():.6f}")


if __name__ == "__main__":
    main()
