def get_multitask_loss(*losses):
    L_total = 0
    for loss in losses:
        L_total += loss / loss.detach()
    return L_total


def get_multitask_loss_v2(iontype_loss, pred_loss):
    return iontype_loss * 0.01 + pred_loss