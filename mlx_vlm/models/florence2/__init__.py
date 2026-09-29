raise RuntimeError("CI fixture: Florence2 import failure one")

import mlx_vlm.models.florence2.processing_florence2  # noqa: F401 (installs processor patch)

from .config import ModelConfig, TextConfig, VisionConfig
from .florence2 import LanguageModel, Model, VisionModel
