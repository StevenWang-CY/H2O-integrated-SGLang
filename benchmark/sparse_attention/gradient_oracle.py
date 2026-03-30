"""Gradient-based page importance oracle.

Replaces attention-weight oracle (which OOMs at 17K tokens — 37GB per layer)
with gradient-based importance: dlogit/dkeys, which is O(seq_len) memory.

The gradient magnitude at each key position indicates how much that position
influences the model's prediction — a direct measure of importance.

Memory budget: bf16 model (~8GB) + activations (~4GB at 8K tokens) + gradients
(~2GB) = ~14GB. Fits in 16GB but tight. For sequences > 8K, use gradient
checkpointing or cap sequence length.

IMPORTANT: Phase 2 (bf16 + gradient oracle) and Phase 3 (FP8 server) CANNOT
run simultaneously on a 16GB GPU. Unload the bf16 model before starting the
FP8 server.

Validation: At <=2K tokens, compare gradient oracle against attention-weight
oracle (attention weights fit at 2K). If Spearman rho > 0.8, the gradient
oracle is validated for use at longer sequences.
"""

import torch
from typing import Optional


def compute_page_importance_gradient(
    model,
    inputs: dict,
    page_size: int = 16,
    target_token_idx: int = -1,
    use_gradient_checkpointing: bool = False,
) -> torch.Tensor:
    """Compute page importance via gradient of target logit w.r.t. key cache.

    Instead of extracting O(seq_len^2) attention weights, compute
    d(logit_target) / d(key_vectors) which is O(seq_len) memory.

    The gradient magnitude at each key position indicates how much
    that position influences the model's prediction.

    Args:
        model: The loaded model (bf16, eager attention).
        inputs: Model inputs dict (from processor).
        page_size: Tokens per page.
        target_token_idx: Which token's logits to differentiate.
            -1 = last token (default).
        use_gradient_checkpointing: Enable gradient checkpointing to
            reduce activation memory (slower but uses less VRAM).

    Returns:
        [num_pages] tensor of importance scores (float32, CPU).
    """
    model.eval()

    # Disable parameter gradients — we only need activation gradients via hooks.
    # This saves ~8 GB (no parameter gradient storage for 4B weights).
    model.requires_grad_(False)

    if use_gradient_checkpointing:
        # use_reentrant=False is required to avoid storing full activations
        # (the default use_reentrant=True stores full activations and OOMs)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Collect gradients from k_proj of each layer
    key_grads: dict[int, torch.Tensor] = {}
    hooks = []

    # Qwen3VL: layers live at model.model.language_model.layers
    if hasattr(model.model, "language_model"):
        all_layers = model.model.language_model.layers
    else:
        all_layers = model.model.layers
    for layer_idx, layer in enumerate(all_layers):
        def make_hook(idx):
            def hook_fn(module, grad_input, grad_output):
                # grad_output[0]: [batch, seq_len, kv_heads * head_dim]
                if grad_output[0] is not None:
                    key_grads[idx] = grad_output[0].detach().cpu()
            return hook_fn

        h = layer.self_attn.k_proj.register_full_backward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Intercept embedding layer output to create a grad leaf without changing
    # model inputs. The hook replaces the embedding output with a detached
    # copy that requires grad — backward flows through it without storing
    # parameter gradients (~8 GB saving for 4B model).
    embed_grad_leaf = [None]

    def embed_output_hook(module, input, output):
        leaf = output.detach().requires_grad_(True)
        embed_grad_leaf[0] = leaf
        return leaf  # Model sees this grad-requiring tensor as embedding output

    embed_layer = model.get_input_embeddings()
    embed_hook_handle = embed_layer.register_forward_hook(embed_output_hook)
    hooks.append(embed_hook_handle)

    try:
        with torch.enable_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, target_token_idx, :]  # [vocab]

            # Use the argmax token as the target (what the model would predict)
            target_logit = logits[logits.argmax()]

            # Backward — flows through layers (via embed grad leaf), hooks
            # at k_proj capture the intermediate grad_output
            target_logit.backward()
    finally:
        # Clean up hooks
        for h in hooks:
            h.remove()

    if use_gradient_checkpointing:
        model.gradient_checkpointing_disable()

    # Restore parameter gradients for subsequent use
    model.requires_grad_(True)

    # Aggregate gradients to page importance
    if not key_grads:
        raise RuntimeError("No gradients captured — check model and hook setup")

    seq_len = list(key_grads.values())[0].shape[1]
    num_pages = (seq_len + page_size - 1) // page_size

    page_importance = torch.zeros(num_pages, dtype=torch.float32)

    for layer_idx, grad in key_grads.items():
        # grad: [1, seq_len, kv_heads * head_dim]
        # L2 norm per position -> [seq_len]
        token_importance = grad[0].float().norm(dim=-1)

        for p in range(num_pages):
            start = p * page_size
            end = min(start + page_size, seq_len)
            page_score = token_importance[start:end].max().item()
            # Max across all layers (any layer finding it important = important)
            page_importance[p] = max(page_importance[p].item(), page_score)

    # Clean up
    model.zero_grad()

    return page_importance


def validate_gradient_oracle(
    model,
    inputs: dict,
    page_size: int = 16,
    max_seq_len: int = 2048,
) -> Optional[float]:
    """Validate gradient oracle against attention-weight oracle at short sequences.

    At <=2K tokens, attention weights fit in memory, so we can compare
    both methods. Returns Spearman correlation coefficient, or None if
    validation couldn't run.

    Args:
        model: Loaded model with eager attention.
        inputs: Model inputs dict (should produce <=max_seq_len tokens).
        page_size: Tokens per page.
        max_seq_len: Max sequence length for validation.

    Returns:
        Spearman rho between gradient and attention-weight page rankings,
        or None if validation failed.
    """
    from scipy.stats import spearmanr

    seq_len = inputs["input_ids"].shape[1]
    if seq_len > max_seq_len:
        print(f"  Sequence too long for validation ({seq_len} > {max_seq_len})")
        return None

    # 1. Gradient oracle
    gradient_importance = compute_page_importance_gradient(model, inputs, page_size)

    # 2. Attention-weight oracle (only feasible at short sequences)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # outputs.attentions: tuple of [batch, heads, seq_len, seq_len] per layer
    num_pages = gradient_importance.shape[0]
    attn_importance = torch.zeros(num_pages, dtype=torch.float32)

    for layer_attn in outputs.attentions:
        # layer_attn: [1, heads, seq_len, seq_len]
        # Average over heads, look at last token's attention distribution
        attn_last = layer_attn[0, :, -1, :].mean(dim=0)  # [seq_len]

        for p in range(num_pages):
            start = p * page_size
            end = min(start + page_size, seq_len)
            page_score = attn_last[start:end].max().item()
            attn_importance[p] = max(attn_importance[p].item(), page_score)

    # 3. Compute Spearman correlation
    rho, pvalue = spearmanr(
        gradient_importance.numpy(), attn_importance.numpy()
    )

    print(f"  Gradient vs Attention-weight oracle: Spearman rho={rho:.3f}, p={pvalue:.4f}")
    if rho > 0.8:
        print(f"  -> VALIDATED (rho > 0.8)")
    else:
        print(f"  -> WARNING: Low correlation (rho <= 0.8)")

    return rho
