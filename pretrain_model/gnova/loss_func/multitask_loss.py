def get_multitask_loss(*losses):
    L_total = 0
    for loss in losses:
        L_total += loss / loss.detach()
    return L_total


def get_multitask_loss_v2(iontype_loss, ionsource_loss):
    # return iontype_loss / iontype_loss.detach() + \
    #     ionsource_loss / ionsource_loss.detach() + \
    #     main_loss / main_loss.detach()
    return iontype_loss + 0.5 * ionsource_loss