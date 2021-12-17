import torch
import torch.nn as nn

# class LabelSmoothing(nn.Module):
#     """
#     NLL loss with label smoothing.
#     """
#     def __init__(self, smoothing=0.0):
#         """
#         Constructor for the LabelSmoothing module.
#         :param smoothing: label smoothing factor
#         """
#         super(LabelSmoothing, self).__init__()
#         self.confidence = 1.0 - smoothing
#         self.smoothing = smoothing
#
#     def forward(self, x, target):
#         logprobs = torch.nn.functional.log_softmax(x, dim=-1)
#
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = self.confidence * nll_loss + self.smoothing * smooth_loss
#         return loss.mean()

class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, smoothing=0.1):
        import torch.nn.functional as F
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (
            1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()
        return loss
