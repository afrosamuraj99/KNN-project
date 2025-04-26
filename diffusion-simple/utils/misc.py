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
    epoch = -1
    for chkpt_path in checkpoints_folder.iterdir():
        parts = chkpt_path.stem.split("-", 1)
        assert len(parts) == 2
        chkpt_epoch = int(parts[1])
        if chkpt_epoch > epoch:
            epoch = chkpt_epoch
            latest_checkpoint_path = chkpt_path
    return latest_checkpoint_path
