def lr_update(
    num_updates: int,
    warmup_updates: int,
    warmup_init_lr: float,
    lr_step: float,
    decay_factor: float,
) -> float:
    """InverseSquareRootSchedule.

    https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/inverse_square_root_schedule.py#L32

    Args:
        num_updates: number of batches already used.
        warmup_updates: number of batch steps for warm up.
        warmup_init_lr: initial learning rate.
        lr_step: step for increasing learning rate during warm up.
        decay_factor: factor for decreasing learning rate after warm up.

    Returns:
        learning rate multiplicate factor
    """
    if num_updates < warmup_updates:
        lr = warmup_init_lr + num_updates * lr_step
    else:
        lr = decay_factor * num_updates ** -0.5
    if warmup_init_lr > 0:
        return lr / warmup_init_lr

    return 0
