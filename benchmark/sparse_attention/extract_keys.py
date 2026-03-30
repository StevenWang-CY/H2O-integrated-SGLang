"""Key vector extraction via forward hooks on bf16 model.

Memory-efficient alternative to output_attentions=True which OOMs at 17K tokens
(37 GB per layer for attention weights). This extracts key vectors at O(seq_len)
memory per layer.

Memory estimate: 17K tokens * 8 kv_heads * 128 head_dim * 2B bf16 * 36 layers
= ~1.25 GB total (vs 37 GB for a SINGLE layer of attention weights).

RoPE caveat: Keys captured from k_proj are PRE-rotation. The actual SGLang KV
cache stores post-rotation keys (RoPE applied inside the attention module after
k_proj). For Quest-vs-H2OQuest comparison, both algorithms see the same
pre-rotation keys in offline mode, so the relative comparison is valid. For
gradient oracle comparison (B1), the oracle also operates on the same pre-rotation
keys, so oracle-vs-algorithm recall is internally consistent.

Usage:
    from extract_keys import KeyVectorExtractor, keys_to_page_bounds, extract_last_query
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to("cuda")

    extractor = KeyVectorExtractor(model)
    keys_per_layer = extractor.extract(inputs)  # {layer_idx: [seq_len, kv_heads, head_dim]}
    k_min, k_max = keys_to_page_bounds(keys_per_layer[17], page_size=16)
"""

import torch
from typing import Optional


class KeyVectorExtractor:
    """Extract key vectors from each layer via forward hooks on k_proj.

    Captures the output of each layer's key projection, which gives us
    the pre-rotation key vectors. These are reshaped from
    [batch, seq_len, kv_heads * head_dim] to [seq_len, kv_heads, head_dim]
    and stored on CPU to minimize GPU memory usage.

    Can also capture last-token query from a specific layer's q_proj in the
    same forward pass (I2 fix: avoids a separate forward pass).
    """

    def __init__(
        self,
        model,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        num_q_heads: int = 32,
        store_on_cpu: bool = True,
    ):
        self.model = model
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.store_on_cpu = store_on_cpu
        self.captured_keys: dict[int, torch.Tensor] = {}
        self.captured_query: Optional[torch.Tensor] = None
        self._hooks: list = []

    def _capture_key(self, layer_idx: int, output: torch.Tensor):
        """Hook callback: capture and reshape key projection output."""
        k = output[0]  # [seq_len, kv_heads * head_dim]
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        if self.store_on_cpu:
            k = k.detach().cpu()
        else:
            k = k.detach()
        self.captured_keys[layer_idx] = k

    def _capture_query(self, module, input, output):
        """Hook callback: capture last-token query projection output."""
        q = output[0, -1, :]  # [q_heads * head_dim]
        q = q.view(self.num_q_heads, self.head_dim)
        self.captured_query = q.detach().cpu()

    def register_hooks(self, query_layer_idx: Optional[int] = None):
        """Register forward hooks on all layers' k_proj modules.

        Args:
            query_layer_idx: If provided, also hook q_proj at this layer
                to capture the last-token query vector in the same pass.
        """
        self.remove_hooks()
        # Qwen3VL: layers live at model.model.language_model.layers
        # Fallback to model.model.layers for other architectures
        if hasattr(self.model.model, "language_model"):
            all_layers = self.model.model.language_model.layers
        else:
            all_layers = self.model.model.layers
        for layer_idx, layer in enumerate(all_layers):
            hook = layer.self_attn.k_proj.register_forward_hook(
                lambda mod, inp, out, idx=layer_idx: self._capture_key(idx, out)
            )
            self._hooks.append(hook)

            if query_layer_idx is not None and layer_idx == query_layer_idx:
                q_hook = layer.self_attn.q_proj.register_forward_hook(
                    self._capture_query
                )
                self._hooks.append(q_hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def extract(
        self,
        inputs: dict,
        query_layer_idx: Optional[int] = None,
    ) -> dict[int, torch.Tensor]:
        """Run a single forward pass and return captured keys per layer.

        If query_layer_idx is provided, the last-token query is also captured
        and available via self.captured_query (as [1, q_heads, head_dim]).

        Args:
            inputs: Model inputs dict (from processor, with input_ids etc).
            query_layer_idx: If provided, also capture last-token query from
                this layer's q_proj (avoids a second forward pass).

        Returns:
            Dict mapping layer_idx -> [seq_len, kv_heads, head_dim] tensor.
        """
        self.captured_keys.clear()
        self.captured_query = None
        self.register_hooks(query_layer_idx=query_layer_idx)
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            self.remove_hooks()
        return dict(self.captured_keys)

    def get_last_query(self) -> Optional[torch.Tensor]:
        """Return captured last-token query as [1, q_heads, head_dim], or None."""
        if self.captured_query is None:
            return None
        return self.captured_query.unsqueeze(0)


class QueryExtractor:
    """Extract the last-token query vector from a specific layer."""

    def __init__(
        self,
        model,
        layer_idx: int,
        num_q_heads: int = 32,
        head_dim: int = 128,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.captured_query: Optional[torch.Tensor] = None
        self._hook = None

    def _capture_query(self, module, input, output):
        """Hook callback: capture last-token query projection output."""
        # output: [batch, seq_len, q_heads * head_dim]
        q = output[0, -1, :]  # [q_heads * head_dim] — last token only
        q = q.view(self.num_q_heads, self.head_dim)  # [q_heads, head_dim]
        self.captured_query = q.detach().cpu()

    def extract(self, inputs: dict) -> torch.Tensor:
        """Run forward pass and return last-token query vector.

        Args:
            inputs: Model inputs dict.

        Returns:
            [1, q_heads, head_dim] tensor (batch dim added for scoring adapter).
        """
        self.captured_query = None
        if hasattr(self.model.model, "language_model"):
            layer = self.model.model.language_model.layers[self.layer_idx]
        else:
            layer = self.model.model.layers[self.layer_idx]
        self._hook = layer.self_attn.q_proj.register_forward_hook(self._capture_query)
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            self._hook.remove()
            self._hook = None
        return self.captured_query.unsqueeze(0)  # [1, q_heads, head_dim]


def extract_last_query(
    model,
    inputs: dict,
    layer_idx: int = 17,
    num_q_heads: int = 32,
    head_dim: int = 128,
) -> torch.Tensor:
    """Convenience function: extract last-token query from one layer.

    Args:
        model: The loaded model.
        inputs: Model inputs dict.
        layer_idx: Which layer to extract from (default 17 = middle layer).
        num_q_heads: Number of query heads.
        head_dim: Head dimension.

    Returns:
        [1, q_heads, head_dim] tensor.
    """
    extractor = QueryExtractor(model, layer_idx, num_q_heads, head_dim)
    return extractor.extract(inputs)


def keys_to_page_bounds(
    keys: torch.Tensor,
    page_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert key vectors to page-level min/max bounds.

    Args:
        keys: [seq_len, kv_heads, head_dim] key vectors from one layer.
        page_size: Number of tokens per page.

    Returns:
        k_min: [num_pages, kv_heads, head_dim] per-page key minimums.
        k_max: [num_pages, kv_heads, head_dim] per-page key maximums.
    """
    seq_len = keys.shape[0]
    num_pages = (seq_len + page_size - 1) // page_size

    # Pad to multiple of page_size (replicate last token)
    if seq_len % page_size != 0:
        pad_len = page_size - (seq_len % page_size)
        keys = torch.cat([keys, keys[-1:].expand(pad_len, -1, -1)], dim=0)

    # Reshape to [num_pages, page_size, kv_heads, head_dim]
    paged = keys.view(num_pages, page_size, keys.shape[1], keys.shape[2])

    k_min = paged.min(dim=1).values  # [num_pages, kv_heads, head_dim]
    k_max = paged.max(dim=1).values

    return k_min, k_max
