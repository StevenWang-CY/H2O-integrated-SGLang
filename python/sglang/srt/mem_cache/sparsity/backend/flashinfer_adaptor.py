"""
FlashInfer sparse backend adaptor.

Adapts sparse page selection results into FlashInfer-compatible kv_indices
and kv_indptr format, then re-plans the decode wrapper with the sparse index set.

Unlike FlashAttentionAdaptor (which rewrites a page_table on a metadata object),
this adaptor directly modifies the FlashInfer wrapper's internal buffers by
calling begin_forward() again with the sparse indices.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import BackendAdaptor

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class FlashInferSparseAdaptor(BackendAdaptor):
    """Adaptor for FlashInfer backend sparse page selection.

    Instead of modifying a metadata object (like FlashAttentionAdaptor), this
    adaptor builds sparse kv_indices/kv_indptr and signals the FlashInfer
    backend to re-plan the decode wrapper before the attention kernel runs.

    The re-plan result is stored as a pending action. The FlashInfer backend's
    forward_decode() checks for pending sparse state and applies it.
    """

    def __init__(self, device: torch.device):
        super().__init__(device)
        # Pending sparse indices for the current layer
        self._pending_sparse = None
        # Original indices for restore after sparse attention
        self._original_state = None

    def save_original_metadata(self, metadata: Any) -> None:
        """No-op for FlashInfer — we save/restore at the wrapper level."""
        pass

    def adapt_for_attn_metadata(
        self,
        selected_indices: torch.Tensor,
        valid_lengths: torch.Tensor,
        sparse_mask: torch.Tensor,
        current_metadata: Any,
        forward_batch: "ForwardBatch",
        req_to_token: torch.Tensor,
        page_size: int,
        layer_id: int,
        **kwargs,
    ) -> Any:
        """Build sparse kv_indices and kv_indptr from selected logical pages.

        Does NOT directly modify the wrapper. Instead, stores the sparse indices
        as pending state. The FlashInfer backend's forward_decode() will read
        this and call begin_forward() with the sparse indices.

        Returns: current_metadata unchanged (FlashInfer ignores the return value).
        """
        if not sparse_mask.any():
            self._pending_sparse = None
            return current_metadata

        bs = selected_indices.shape[0]

        # Build sparse kv_indices from selected logical page indices
        # selected_indices: [bs, max_selected] — logical page indices per request
        # valid_lengths: [bs] — how many selected pages per request
        max_selected = selected_indices.shape[1]

        # Convert logical page indices to physical token indices
        # Each logical page i starts at token position i * page_size
        page_starts = selected_indices * page_size  # [bs, max_selected]
        page_starts_clamped = page_starts.clamp(min=0, max=req_to_token.shape[1] - 1)

        # Look up physical token indices for the first token of each selected page
        req_indices_expanded = forward_batch.req_pool_indices.unsqueeze(1).expand(
            -1, max_selected
        )
        first_tokens = req_to_token[req_indices_expanded, page_starts_clamped]

        # Physical page index = physical_token // page_size
        physical_pages = (first_tokens // page_size).to(torch.int32)

        # Mask out invalid entries
        valid_mask = torch.arange(max_selected, device=self.device).unsqueeze(0) < valid_lengths.unsqueeze(1)
        # For invalid entries, set to 0 (won't be read due to indptr)
        physical_pages = torch.where(
            valid_mask & sparse_mask.unsqueeze(1),
            physical_pages,
            torch.zeros_like(physical_pages),
        )

        # Build sparse kv_indptr (CSR format)
        sparse_lens = torch.where(sparse_mask, valid_lengths, forward_batch.seq_lens // page_size + 1).to(torch.int32)
        sparse_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        sparse_indptr[1:] = torch.cumsum(sparse_lens, dim=0)

        # Build flat kv_indices array
        total_sparse_pages = sparse_indptr[-1].item()
        sparse_kv_indices = torch.zeros(
            total_sparse_pages, dtype=torch.int32, device=self.device
        )

        # Pack selected physical pages into flat array
        for i in range(bs):
            if sparse_mask[i]:
                n_pages = valid_lengths[i].item()
                start = sparse_indptr[i].item()
                sparse_kv_indices[start : start + n_pages] = physical_pages[i, :n_pages]

        # Compute kv_last_page_len for sparse selection
        seq_lens = forward_batch.seq_lens
        sparse_last_page_len = ((seq_lens - 1) % page_size + 1).to(torch.int32)

        self._pending_sparse = {
            "kv_indices": sparse_kv_indices,
            "kv_indptr": sparse_indptr,
            "kv_last_page_len": sparse_last_page_len,
            "sparse_lens": sparse_lens,
            "sparse_mask": sparse_mask,
        }

        return current_metadata

    def get_pending_sparse(self):
        """Get and clear pending sparse state. Called by FlashInfer backend."""
        state = self._pending_sparse
        self._pending_sparse = None
        return state

    def has_pending_sparse(self):
        """Check if there's a pending sparse re-plan."""
        return self._pending_sparse is not None
