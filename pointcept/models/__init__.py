from .builder import build_model
from .modules import PointModule, PointModel

try:
    from .default import DefaultSegmentor, DefaultClassifier
except ImportError:
    pass

# Backbones (motivation only needs volt/sonata/voltsonata)
try:
    from .sparse_unet import *
except ImportError:
    pass
try:
    from .point_transformer import *
except ImportError:
    pass
try:
    from .point_transformer_v2 import *
except ImportError:
    pass
try:
    from .point_transformer_v3 import *
except ImportError:
    pass
try:
    from .stratified_transformer import *
except ImportError:
    pass
try:
    from .spvcnn import *
except ImportError:
    pass
try:
    from .octformer import *
except ImportError:
    pass
try:
    from .oacnns import *
except ImportError:
    pass
from .volt import *

# from .swin3d import *

# Semantic Segmentation
try:
    from .context_aware_classifier import *
except ImportError:
    pass

# Instance Segmentation
try:
    from .point_group import *
except ImportError:
    pass
try:
    from .sgiformer import *
except ImportError:
    pass
try:
    from .spformer import *
except ImportError:
    pass

# Pretraining
try:
    from .masked_scene_contrast import *
except ImportError:
    pass
try:
    from .point_prompt_training import *
except ImportError:
    pass
from .sonata import *
try:
    from .concerto import *
except ImportError:
    pass
from .voltsonata import *
