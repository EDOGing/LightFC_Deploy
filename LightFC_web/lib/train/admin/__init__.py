from importlib.util import find_spec

from .environment import env_settings, create_default_local_file_ITP_train
from .stats import AverageMeter, StatValue

# TensorBoard is a training-only dependency.  Keep model/checkpoint loading
# usable in lightweight inference environments where it is not installed.
if find_spec("tensorboard") is not None or find_spec("tensorboardX") is not None:
    from .tensorboard import TensorboardWriter
else:
    TensorboardWriter = None
