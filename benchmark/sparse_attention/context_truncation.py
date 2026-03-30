"""Context truncation for action quality evaluation.

Simulates sparse attention by rebuilding prompts using only tokens from
selected pages. This is STRICTER than real sparse attention (where non-selected
tokens still exist in the KV cache and partially contribute via sink/recent
pages). If H2OQuest-truncated outperforms Quest-truncated under this stricter
constraint, the real advantage with actual sparse attention would be even larger.

Usage:
    from context_truncation import truncate_context_to_pages, run_context_truncation_comparison
"""

import base64
from typing import Optional


def truncate_context_to_pages(
    full_html: str,
    selected_pages: set[int],
    page_size: int,
    tokenizer,
) -> str:
    """Rebuild HTML context keeping only tokens from selected pages.

    Tokenizes the full HTML, collects tokens from selected page indices,
    and decodes back to text.

    Args:
        full_html: Full HTML string.
        selected_pages: Set of page indices to keep.
        page_size: Tokens per page.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Truncated HTML string containing only tokens from selected pages.
    """
    tokens = tokenizer.encode(full_html, add_special_tokens=False)
    num_pages = (len(tokens) + page_size - 1) // page_size

    selected_tokens = []
    for page_idx in sorted(selected_pages):
        if page_idx >= num_pages:
            continue
        start = page_idx * page_size
        end = min(start + page_size, len(tokens))
        selected_tokens.extend(tokens[start:end])

    truncated_html = tokenizer.decode(selected_tokens, skip_special_tokens=False)
    return truncated_html


def evaluate_action(
    predicted: str,
    ground_truth_op: str,
    ground_truth_value: str,
    target_action_repr: str,
) -> dict:
    """Evaluate predicted action against Mind2Web ground truth.

    Args:
        predicted: Model's predicted action string.
        ground_truth_op: Ground truth operation type (e.g., "CLICK", "TYPE").
        ground_truth_value: Ground truth value for the operation.
        target_action_repr: Human-readable action description
            (e.g., "Click on 'Search' button").

    Returns:
        Dict with operation_match, element_match, exact_match booleans.
    """
    pred_upper = predicted.upper().strip()

    # Operation match: ground truth op type appears in prediction
    op_match = ground_truth_op.upper() in pred_upper

    # Element match: key terms from target_action_repr appear in prediction
    key_terms = _extract_key_terms(target_action_repr)
    element_match = any(term.lower() in predicted.lower() for term in key_terms)

    return {
        "operation_match": op_match,
        "element_match": element_match,
        "exact_match": op_match and element_match,
    }


def _extract_key_terms(action_repr: str) -> list[str]:
    """Extract key terms from action representation for element matching.

    E.g., "Click on 'Search' button" -> ["Search", "button"]
    """
    import re

    terms = []

    # Extract quoted strings
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", action_repr)
    terms.extend(quoted)

    # Extract nouns/descriptors (words after "on", "the", etc.)
    # Simple heuristic: take capitalized words and words after prepositions
    words = action_repr.split()
    skip_words = {
        "click", "type", "select", "on", "the", "a", "an", "in", "into",
        "with", "for", "to", "at", "of", "and", "or",
    }
    for word in words:
        cleaned = word.strip(".,;:'\"()[]{}!?")
        if cleaned and cleaned.lower() not in skip_words and len(cleaned) > 2:
            if cleaned not in terms:
                terms.append(cleaned)

    return terms if terms else [action_repr]


def run_context_truncation_comparison(
    server_url: str,
    episode: dict,
    quest_selections_per_step: list[set[int]],
    h2oquest_selections_per_step: list[set[int]],
    tokenizer,
    page_size: int = 16,
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    max_html_chars: int = 8000,
) -> list[dict]:
    """Compare action quality across dense, Quest-truncated, H2OQuest-truncated.

    Sends requests to the FP8 FlashInfer server via OpenAI API.

    Args:
        server_url: FP8 server URL (e.g., "http://localhost:30000").
        episode: Dict with keys: task, name, steps (list of step dicts).
            Each step has: html, screenshot_path (optional), action_repr,
            operation, value, is_return_step (optional).
        quest_selections_per_step: Quest page selections per step.
        h2oquest_selections_per_step: H2OQuest page selections per step.
        tokenizer: HuggingFace tokenizer for truncation.
        page_size: Tokens per page.
        model_name: Model name for OpenAI API.
        max_html_chars: Max HTML characters in prompt (safety truncation).

    Returns:
        List of result dicts with: episode, step, method, action,
        html_tokens, is_return_step.
    """
    import openai

    client = openai.Client(base_url=f"{server_url}/v1", api_key="EMPTY")
    results = []

    for step_idx, step in enumerate(episode["steps"]):
        # Build action history
        action_history_text = "\n".join(
            f"Step {i + 1}: {s['action_repr']}"
            for i, s in enumerate(episode["steps"][:step_idx])
        )

        # Three versions of HTML context
        full_html = step["html"]

        quest_html = truncate_context_to_pages(
            full_html,
            quest_selections_per_step[step_idx],
            page_size,
            tokenizer,
        )
        h2oquest_html = truncate_context_to_pages(
            full_html,
            h2oquest_selections_per_step[step_idx],
            page_size,
            tokenizer,
        )

        # Build message content (with optional screenshot)
        for label, html_context in [
            ("dense", full_html),
            ("quest", quest_html),
            ("h2oquest", h2oquest_html),
        ]:
            prompt_text = (
                f"Task: {episode['task']}\n"
                f"Previous actions:\n{action_history_text}\n\n"
                f"Current page HTML:\n{html_context[:max_html_chars]}\n\n"
                f"What is the next action? Respond with exactly one of:\n"
                f"CLICK(element_description)\n"
                f"TYPE(element_description, text)\n"
                f"SELECT(element_description, value)"
            )

            content = []

            # Add screenshot if available
            screenshot_path = step.get("screenshot_path")
            if screenshot_path:
                try:
                    with open(screenshot_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                    })
                except (FileNotFoundError, IOError):
                    pass  # Skip screenshot if not available

            content.append({"type": "text", "text": prompt_text})

            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=64,
                    temperature=0.0,
                )
                action = response.choices[0].message.content
            except Exception as e:
                action = f"ERROR: {e}"

            results.append({
                "episode": episode["name"],
                "step": step_idx,
                "method": label,
                "action": action,
                "html_tokens": len(tokenizer.encode(html_context)),
                "is_return_step": step.get("is_return_step", False),
            })

    return results
