# SPDX-License-Identifier: Apache-2.0
"""Bernini MLLM custom attention mask (ported from Bernini attention_utils.py)."""

import torch


def build_custom_attention_mask(token_type, token_segment_ids):
    """Build Bernini planner attention mask.

    Args:
        token_type: (B, L) with values 0(t), 1(p), 2(i), 3(o).
        token_segment_ids: (B, L) segment ids.

    Returns:
        (B, L, L) float mask — visible=0.0, masked=-inf.
    """
    B, L = token_type.shape
    device = token_type.device

    q_type = token_type.unsqueeze(2)
    k_type = token_type.unsqueeze(1)
    q_id = token_segment_ids.unsqueeze(2)
    k_id = token_segment_ids.unsqueeze(1)

    causal_mask = torch.tril(torch.ones((L, L), device=device, dtype=torch.bool)).unsqueeze(0)

    k_is_ti = (k_type == 0) | (k_type == 2)
    k_is_p = k_type == 1
    k_is_o = k_type == 3
    ids_match = q_id == k_id

    visible_base_ti = causal_mask & k_is_ti
    visible_p_bidirectional = k_is_p & ids_match
    visible_o_bidirectional = k_is_o & ids_match

    final_bool_mask = torch.zeros((B, L, L), device=device, dtype=torch.bool)

    q_is_ti = (q_type == 0) | (q_type == 2)
    final_bool_mask = final_bool_mask | (q_is_ti & visible_base_ti)

    q_is_p = q_type == 1
    final_bool_mask = final_bool_mask | (q_is_p & (visible_base_ti | visible_p_bidirectional))

    q_is_o = q_type == 3
    final_bool_mask = final_bool_mask | (q_is_o & (visible_base_ti | visible_o_bidirectional))

    attention_mask = torch.zeros((B, L, L), device=device, dtype=torch.float32)
    attention_mask.masked_fill_(~final_bool_mask, float("-inf"))
    return attention_mask
