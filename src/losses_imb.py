import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


def focal_loss(logits, labels, alpha=None, gamma=2):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bc_loss = F.binary_cross_entropy(input=logits, target=labels, reduction="none")
    alpha = alpha[labels.long()].cuda()

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bc_loss

    if alpha is not None:
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
    else:
        focal_loss = torch.sum(loss)

    focal_loss /= labels.shape[0]
    return focal_loss


class Loss_imb(torch.nn.Module):
    def __init__(
        self,
        loss_type: str = "binary_cross_entropy",
        beta: float = 0.999,
        fl_gamma=2,
        samples_per_class=None,
        samples_weight=None,
        class_balanced=False,
        paper_version=False,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        reference: https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
        Args:
            loss_type: string. One of "focal_loss", "cross_entropy",
                "binary_cross_entropy", "softmax_binary_cross_entropy".
            beta: float. Hyperparameter for Class balanced loss.
            fl_gamma: float. Hyperparameter for Focal loss.
            samples_per_class: A python list of size [num_classes].
                Required if class_balance is True.
            class_balanced: bool. Whether to use class balanced loss.
        Returns:
            Loss instance
        """
        super(Loss_imb, self).__init__()

        # if class_balanced is True and samples_per_class is None:
        #     raise ValueError("samples_per_class cannot be None when class_balanced is True")

        self.loss_type = loss_type
        self.beta = beta
        self.fl_gamma = fl_gamma
        self.samples_per_class = samples_per_class
        self.class_balanced = class_balanced
        self.samples_weight = samples_weight
        self.paper_version = paper_version

    def forward(
        self,
        logits: torch.tensor,
        labels: torch.tensor,
    ):
        """
        Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
        Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
        where Loss is one of the standard losses used for Neural Networks.
        Args:
            logits: A float tensor of size [batch, num_classes].
            labels: An int tensor of size [batch].
        Returns:
            cb_loss: A float tensor representing class balanced loss
        """

        # batch_size = logits.size(0)
        num_classes = 2
        weights = self.samples_weight
        w_ex = weights[labels.cpu().long()].cuda()
        cb_loss = F.binary_cross_entropy(logits, labels, weight=w_ex)

        # inputs = torch.tensor([0.8, 0.2], dtype=torch.float16).cuda()
        # labels = torch.tensor([1.0, 0.0], dtype=torch.float16).cuda()
        #
        # # Instantiate the BCELoss object
        # loss_fn = nn.BCELoss()
        #
        # # Compute the binary cross entropy loss
        # loss = loss_fn(inputs, labels)
        #
        # inputs = torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=torch.float16).cuda()
        # labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float16).cuda()
        # weights = torch.tensor([2.0, 1.0], dtype=torch.float16).cuda()
        # loss = F.binary_cross_entropy(inputs, labels, weight=weights)

        # if self.samples_per_class is not None:
        #     if self.paper_version:
        #         effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
        #         weights = (1.0 - self.beta) / np.array(effective_num)
        #         weights = weights / np.sum(weights) * num_classes
        #         weights = torch.tensor(weights, device=logits.device).float()
        #     else:
        #         weights = self.samples_weight
        #
        #     if self.loss_type == "focal_loss":
        #         cb_loss = focal_loss(logits, labels, alpha=weights, gamma=self.fl_gamma)
        #     elif self.loss_type == "binary_cross_entropy":
        #         w_ex = weights[labels.long()].cuda()
        #         cb_loss = F.binary_cross_entropy(logits, labels, weight=w_ex)
        # else:
        #     cb_loss = F.binary_cross_entropy(logits, labels)
        return cb_loss