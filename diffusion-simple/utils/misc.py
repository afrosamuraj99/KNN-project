from inspect import isfunction

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def transform(image):
    return ((image / 255) * 2) - 1

def reverse_transform(image):
    return ((image + 1) / 2) * 255

def latest_checkpoint(checkpoints_folder):
    latest_checkpoint_path = None
    latest_ema_checkpoint_path = None
    latest_opt_checkpoint_path = None

    if not checkpoints_folder.exists():
        return latest_checkpoint_path, latest_ema_checkpoint_path, latest_opt_checkpoint_path

    epoch = -1
    ema_epoch = -1
    opt_epoch = -1

    for chkpt_path in checkpoints_folder.iterdir():
        parts = chkpt_path.stem.split("-", 2)
        assert len(parts) == 3
        chkpt_epoch = int(parts[1])

        if parts[2] == "model":
            if chkpt_epoch > epoch:
                epoch = chkpt_epoch
                latest_checkpoint_path = chkpt_path
        elif parts[2] == "ema":
            if chkpt_epoch > ema_epoch:
                ema_epoch = chkpt_epoch
                latest_ema_checkpoint_path = chkpt_path
        elif parts[2] == "optimizer":
            if chkpt_epoch > opt_epoch:
                opt_epoch = chkpt_epoch
                latest_opt_checkpoint_path = chkpt_path

    assert epoch == ema_epoch == opt_epoch
    return latest_checkpoint_path, latest_ema_checkpoint_path, latest_opt_checkpoint_path

def epoch_checkpoint(checkpoints_folder, epoch):
    checkpoint_path = None
    ema_checkpoint_path = None
    opt_chechkpoint_path = None

    if not checkpoints_folder.exists():
        return checkpoint_path, ema_checkpoint_path, opt_chechkpoint_path

    for chkpt_path in checkpoints_folder.iterdir():
        parts = chkpt_path.stem.split("-", 2)
        assert len(parts) == 3
        chkpt_epoch = int(parts[1])

        if chkpt_epoch == epoch:
            if parts[2] == "model":
                checkpoint_path = chkpt_path
            elif parts[2] == "ema":
                ema_checkpoint_path = chkpt_path
            elif parts[2] == "optimizer":
                opt_checkpoint_path = chkpt_path

        if checkpoint_path is not None and ema_checkpoint_path is not None and opt_checkpoint_path is not None:
            break

    assert checkpoint_path is not None and ema_checkpoint_path is not None and opt_checkpoint_path is not None
    return checkpoint_path, ema_checkpoint_path, opt_chechkpoint_path
