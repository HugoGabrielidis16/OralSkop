"""Model package for the custom torchseg path.

Re-exports the factory API so existing imports keep working:
``from oralskop.torchseg.model import build_model, has_aux``.
"""

from oralskop.torchseg.model.factory import build_model, has_aux
from oralskop.torchseg.model.unet import UNet

__all__ = ["build_model", "has_aux", "UNet"]
