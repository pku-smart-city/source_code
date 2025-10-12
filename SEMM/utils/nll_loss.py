import torch

def nb_zeroinflated_nll_loss(y, n, p, pi, y_mask=None):
    """
    y: true values
    y_mask: whether missing mask is given
    """
    idx_yeq0 = y == 0
    idx_yg0 = y > 0

    n_yeq0 = n[idx_yeq0]
    p_yeq0 = p[idx_yeq0]
    pi_yeq0 = pi[idx_yeq0]
    yeq0 = y[idx_yeq0]

    n_yg0 = n[idx_yg0]
    p_yg0 = p[idx_yg0]
    pi_yg0 = pi[idx_yg0]
    yg0 = y[idx_yg0]

    L_yeq0 = torch.log(pi_yeq0) + torch.log((1 - pi_yeq0) * torch.pow(p_yeq0, n_yeq0))
    L_yg0 = torch.log(1 - pi_yg0) + torch.lgamma(n_yg0 + yg0) - torch.lgamma(yg0 + 1) - torch.lgamma(n_yg0) + n_yg0 * torch.log(p_yg0) + yg0 * torch.log(1 - p_yg0)

    return -torch.sum(L_yeq0) - torch.sum(L_yg0)