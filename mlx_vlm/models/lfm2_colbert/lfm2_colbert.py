from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..lfm2_embedding.lfm2_embedding import Model as Lfm2BidirectionalModel
from ..pooling import EmbeddingOutput, normalize_embeddings
from .config import ModelConfig


class Model(Lfm2BidirectionalModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.projection = nn.Linear(
            config.hidden_size, config.embedding_dim, bias=False
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs,
    ) -> EmbeddingOutput:
        h, _ = self._encode(input_ids, attention_mask, mask_convolutions=False)
        text_embeds = normalize_embeddings(self.projection(h).astype(mx.float32))
        return EmbeddingOutput(last_hidden_state=h, text_embeds=text_embeds)

    def sanitize(self, weights):
        out = super().sanitize(weights)
        for source, target in (
            ("model.1_Dense.linear.weight", "projection.weight"),
            ("model.dense.weight", "projection.weight"),
            ("model.dense.scales", "projection.scales"),
            ("model.dense.biases", "projection.biases"),
        ):
            if source in out:
                out[target] = out.pop(source)
        return out
