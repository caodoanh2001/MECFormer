import torch
import torch.nn as nn
import torch.nn.functional as F

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def seesaw_ce_loss(cls_score,
                   labels,
                   weight,
                   cum_samples,
                   num_classes,
                   p,
                   q,
                   eps,
                   reduction='mean',
                   avg_factor=None,
                   ignore_index=None):
    """Calculate the Seesaw CrossEntropy loss.
    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor): Sample-wise loss weight.
        cum_samples (torch.Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    Returns:
        torch.Tensor: The calculated loss
    """
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes

    onehot_labels = F.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(onehot_labels.size())

    # mitigation factor
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(
            min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index
                                                               )  # M_{ij}
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    # compensation factor
    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[
            torch.arange(0, len(scores)).to(scores.device).long(),
            labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor

    cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))
    loss = F.cross_entropy(F.log_softmax(cls_score, 1), labels, weight=None, reduction='none', ignore_index=ignore_index)

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss

def o_seesaw_ce_loss(cls_score,
                    o_cls_score,
                   labels,
                   weight,
                   cum_samples,
                   num_classes,
                   p,
                   q,
                   k,
                   eps,
                   reduction='mean',
                   avg_factor=None):
    """Calculate the Seesaw CrossEntropy loss.
    Args:
        cls_score (torch.Tensor): The prediction with shape (N, C),
             C is the number of classes.
        labels (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor): Sample-wise loss weight.
        cum_samples (torch.Tensor): Cumulative samples for each category.
        num_classes (int): The number of classes.
        p (float): The ``p`` in the mitigation factor.
        q (float): The ``q`` in the compenstation factor.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    Returns:
        torch.Tensor: The calculated loss
    """
    assert cls_score.size(-1) == num_classes
    assert len(cum_samples) == num_classes

    onehot_labels = F.one_hot(labels, num_classes)
    seesaw_weights = cls_score.new_ones(onehot_labels.size())

    # mitigation factor
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(
            min=1) / cum_samples[:, None].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index
                                                               )  # M_{ij}
        mitigation_factor = sample_weights[labels.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    # compensation factor
    if q > 0:
        scores = F.softmax(cls_score.detach(), dim=1)
        self_scores = scores[
            torch.arange(0, len(scores)).to(scores.device).long(),
            labels.long()]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor

    # Ordinal compensation factor
    if k > 0:
        o_scores = F.softmax(o_cls_score.detach(), dim=1)
        self_scores = o_scores[
            torch.arange(0, len(o_scores)).to(o_scores.device).long(),
            labels.long()]
        ordinal_score_matrix = o_scores / self_scores[:, None].clamp(min=eps)
        index = (ordinal_score_matrix > 1.0).float()
        ordinal_compensation_factor = ordinal_score_matrix.pow(k) * index + (1 - index)
        seesaw_weights = seesaw_weights * ordinal_compensation_factor

    cls_score = cls_score + (seesaw_weights.log() * (1 - onehot_labels))

    loss = F.cross_entropy(cls_score, labels, weight=None, reduction='none')

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss

class SeesawLoss(nn.Module):
    """Implementation of seesaw loss.
    Refers to `Seesaw Loss for Long-Tailed Instance Segmentation (CVPR 2021)
    <https://arxiv.org/abs/2008.10032>`_
    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid of softmax.
             Only False is supported. Defaults to False.
        p (float): The ``p`` in the mitigation factor.
             Defaults to 0.8.
        q (float): The ``q`` in the compenstation factor.
             Defaults to 2.0.
        num_classes (int): The number of classes.
             Default to 1000 for the ImageNet dataset.
        eps (float): The minimal value of divisor to smooth
             the computation of compensation factor, default to 1e-2.
        reduction (str): The method that reduces the loss to a scalar.
             Options are "none", "mean" and "sum". Default to "mean".
        loss_weight (float): The weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_sigmoid=False,
                 p=0.8,
                 q=2.0,
                 num_classes=1000,
                 eps=1e-2,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=None):
        super(SeesawLoss, self).__init__()
        assert not use_sigmoid, '`use_sigmoid` is not supported'
        self.use_sigmoid = False
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

        self.cls_criterion = seesaw_ce_loss

        # cumulative samples for each category
        self.register_buffer('cum_samples',
                             torch.zeros(self.num_classes, dtype=torch.float))

    def forward(self,
                cls_score,
                labels,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The prediction with shape (N, C).
            labels (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                 the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                 Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            f'The `reduction_override` should be one of (None, "none", ' \
            f'"mean", "sum"), but get "{reduction_override}".'
        assert cls_score.size(0) == labels.view(-1).size(0), \
            f'Expected `labels` shape [{cls_score.size(0)}], ' \
            f'but got {list(labels.size())}'
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert cls_score.size(-1) == self.num_classes, \
            f'The channel number of output ({cls_score.size(-1)}) does ' \
            f'not match the `num_classes` of seesaw loss ({self.num_classes}).'

        # accumulate the samples for each category
        unique_labels = labels.unique()
        for u_l in unique_labels:
            inds_ = labels == u_l.item()
            self.cum_samples[u_l] = self.cum_samples[u_l] + inds_.sum()

        if weight is not None:
            weight = weight.float()
        else:
            weight = labels.new_ones(labels.size(), dtype=torch.float)

        # calculate loss_cls_classes
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score, labels, weight, self.cum_samples, self.num_classes,
            self.p, self.q, self.eps, reduction, avg_factor, ignore_index=self.ignore_index)

        return loss_cls