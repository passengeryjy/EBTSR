import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)    #返回一个元组，第一个元素是logits中每行的前num_labels个最大值，第二个元素是每个最大值的列索引（这里用下划线表示不关心这个值）
            top_v = top_v[:, -1]    #取出了每行最大值中的最小值。也就是说，只有比这个最小值大的那些数才会被保留下来。
            mask = (logits >= top_v.unsqueeze(1)) & mask    #原来的mask和新生成的一个mask相乘，生成一个新的mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)#通过判断每个样本在其他列是否有值为 1，如果没有，则说明该样本不是正样本，将其对应的第 0 列赋值为 1。
        return output
