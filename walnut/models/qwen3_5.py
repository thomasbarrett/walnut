"""Qwen3.5 (``Qwen3_5ForConditionalGeneration``).

Composes the primitives in `walnut.layers` into the Qwen3.5 module tree: a
vision encoder plus a hybrid text decoder (interleaved full and linear
attention) and a tied LM head.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from walnut.cache import Batch, Cache, CachePool, CacheView, pages_for
from walnut.layers import (
    Attention,
    FusedLinear,
    GatedDeltaNet,
    RMSNorm,
    RotaryEmbedding,
    VisionAttention,
    VisionPatchEmbed,
    VisionPatchMerger,
    VisionRotaryEmbedding,
    apply_rotary_pos_emb,
)
from walnut.layers.attention import KVCache
from walnut.layers.linear_attention import ConvState
from walnut.models.loader import copy_weights
from walnut.runner.graphs import DecodeGraph
from walnut.runner.sampling import Sampler
from walnut.scheduler.request import SamplingParams


class Qwen3_5MLP(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``.

    ``gate`` and ``up`` read the same input, so they are held as one
    `FusedLinear` — one gemv instead of two, which at decode's batch of 1 is
    the difference between one memory-bound kernel and two.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden, inter = config.hidden_size, config.intermediate_size
        self.gate_up_proj = FusedLinear(hidden, {"gate_proj": inter, "up_proj": inter})
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class Qwen3_5Attention(nn.Module):
    """Full grouped-query self-attention with QK-norm and output gating.

    When ``attn_output_gate`` is set, the query half of ``qkv_proj`` is doubled:
    half is the query, half is a sigmoid gate applied to the attention output
    before ``o_proj``.

    Q, K and V read the same input and are held as one `FusedLinear`, split
    after.
    """

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.output_gate = config.attn_output_gate
        bias = config.attention_bias

        q_out = self.num_heads * self.head_dim * (2 if self.output_gate else 1)
        kv_out = self.num_kv_heads * self.head_dim
        self.qkv_proj = FusedLinear(
            config.hidden_size,
            {"q_proj": q_out, "k_proj": kv_out, "v_proj": kv_out},
            bias=bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.attn = Attention(self.num_heads, self.num_kv_heads, self.head_dim)

    def make_cache(
        self,
        pages: int,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> KVCache:
        return self.attn.make_cache(pages, dtype, device)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache,
        batch: Batch,
    ) -> torch.Tensor:
        bsz, seq, _ = x.shape

        q, k, v = self.qkv_proj(x)

        gate = None
        if self.output_gate:
            # Reshape to (B, S, heads, 2*D) first, so query/gate split per head.
            q = q.reshape(bsz, seq, self.num_heads, 2 * self.head_dim)
            q, gate = q.chunk(2, dim=-1)
            gate = gate.reshape(bsz, seq, -1)
        else:
            q = q.reshape(bsz, seq, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seq, self.num_kv_heads, self.head_dim)
        v = v.reshape(bsz, seq, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn = self.attn(q, k, v, cache, batch).reshape(bsz, seq, -1)
        if gate is not None:
            attn = attn * torch.sigmoid(gate)
        return self.o_proj(attn)


class Qwen3_5DecoderLayer(nn.Module):
    """One text block: a token mixer (full or linear attention) + SwiGLU MLP.

    Per ``config.layer_types[layer_idx]``, full-attention layers hold
    ``self_attn`` and linear-attention layers hold ``linear_attn``.
    """

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_key_heads=config.linear_num_key_heads,
                num_value_heads=config.linear_num_value_heads,
                key_head_dim=config.linear_key_head_dim,
                value_head_dim=config.linear_value_head_dim,
                conv_kernel_dim=config.linear_conv_kernel_dim,
            )
        self.mlp = Qwen3_5MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def make_cache(
        self,
        rows: int,
        pages: int,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> Cache:
        """This layer's share of the pool.

        Two sizes because the two mixers hold two kinds of state: full
        attention takes ``pages`` of the shared key/value pool and no row at
        all, and linear attention takes a row of recurrent state per sequence
        and no pages. See `walnut.cache.state`.
        """
        if self.block_type == "full_attention":
            return self.self_attn.make_cache(pages, dtype, device)
        return self.linear_attn.make_cache(rows, dtype, device)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: Cache,
        batch: Batch,
    ) -> torch.Tensor:
        normed = self.input_layernorm(x)
        if self.block_type == "full_attention":
            assert not isinstance(cache, ConvState)
            x = x + self.self_attn(normed, cos, sin, cache, batch)
        else:
            assert not isinstance(cache, KVCache)
            x = x + self.linear_attn(normed, cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3_5TextModel(nn.Module):
    """Text decoder stack: embeddings, hybrid layers, final norm.

    Built from ``config.text_config``; owns the shared rotary embedding.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        rope = config.rope_parameters
        self.rotary = RotaryEmbedding(
            head_dim=config.head_dim,
            rope_theta=rope["rope_theta"],
            partial_rotary_factor=rope["partial_rotary_factor"],
            mrope_section=rope["mrope_section"],
        )

    def make_cache(
        self,
        max_seq_len: int,
        max_batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        pages: int | None = None,
    ) -> CachePool:
        """Build a fresh per-layer cache (paged KV for full attention, conv +
        recurrent state for linear attention).

        ``max_batch_size`` sets the rows — how many sequences can be in flight
        — and ``pages`` the key/value pool they draw from, defaulting to the
        parity figure where every row could run to ``max_seq_len`` at once.
        ``max_seq_len`` remains the longest single sequence, because it is what
        a row's block table can address.

        A pool rather than a bare list: the placement of sequences within it is
        part of what the cache *is*, and everything above here — a prefill
        against one row, a decode against a bucket, a scheduler handing rows
        out — asks the pool for it rather than working it out again.
        """
        pages = max_batch_size * pages_for(max_seq_len) if pages is None else pages
        return CachePool(
            [
                # One page more than the pool hands out: `CachePool.SCRATCH`,
                # which idle rows write into and no sequence can hold.
                cast(Qwen3_5DecoderLayer, layer).make_cache(
                    max_batch_size, pages + 1, dtype, device
                )
                for layer in self.layers
            ],
            max_batch_size,
            max_seq_len,
            pages,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        *,
        cache: CacheView,
        batch: Batch | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)
        h = inputs_embeds
        if positions is None:
            positions = torch.arange(h.shape[1], device=h.device)

        # Rotary reads the positions as given; the cache reads where they land.
        # One `Batch` for the whole pass, so the layout is resolved once rather
        # than re-derived from the position tensor by every mixer.
        #
        # A caller that already knows the layout passes it: a serving step
        # covers rows sitting at unrelated positions, and soon rows
        # contributing unequal numbers of tokens, neither of which the shape of
        # a position tensor can state. Deriving it here is the fallback for the
        # callers that have nothing else to say — `iter_generate`, and a graph
        # capture, which must build it inside the captured region so the page
        # lookup is replayed rather than frozen.
        cos, sin = self.rotary(positions)
        if batch is None:
            batch = cache.batch(positions)
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, cache[i], batch)
        return self.norm(h)


def _vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor, num_grid_per_side: int, spatial_merge_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear interpolation indices/weights into the pos-embed table:
    (4, total) corner indices and weights, in merge-block order."""
    side = num_grid_per_side
    merge_size = spatial_merge_size
    device = grid_thw.device

    idx_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)

        h_grid = torch.linspace(0, side - 1, h, device=device)
        w_grid = torch.linspace(0, side - 1, w, device=device)

        h_floor = h_grid.int()
        w_floor = w_grid.int()
        h_ceil = (h_floor + 1).clamp(max=side - 1)
        w_ceil = (w_floor + 1).clamp(max=side - 1)

        h_frac = h_grid - h_floor
        w_frac = w_grid - w_floor

        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side

        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            ((1 - h_frac)[:, None] * (1 - w_frac)[None, :]).flatten(),
            ((1 - h_frac)[:, None] * w_frac[None, :]).flatten(),
            (h_frac[:, None] * (1 - w_frac)[None, :]).flatten(),
            (h_frac[:, None] * w_frac[None, :]).flatten(),
        ]

        h_idx = torch.arange(h, device=device).view(h // merge_size, merge_size)
        w_idx = torch.arange(w, device=device).view(w // merge_size, merge_size)
        reorder = (
            (h_idx[:, :, None, None] * w + w_idx[None, None, :, :])
            .transpose(1, 2)
            .flatten()
            .repeat(t)
        )

        for i in range(4):
            idx_parts[i].append(corner_indices[i][reorder])
            weight_parts[i].append(corner_weights[i][reorder])

    bilinear_indices = torch.stack([torch.cat(p) for p in idx_parts])
    bilinear_weights = torch.stack([torch.cat(p) for p in weight_parts])
    return bilinear_indices, bilinear_weights


def _vision_encoder_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> torch.Tensor:
    """2-axis (h, w) rotary position ids for the vision encoder (each patch's
    position within its image), block-major over merge blocks.

    Distinct from Qwen3_5Model.get_vision_position_ids (3-axis, LLM sequence).
    """
    device = grid_thw.device
    position_ids = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos_ids, wpos_ids = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        block_shape = (
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        wpos_ids = wpos_ids.reshape(block_shape).transpose(1, 2).flatten()
        position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
    return torch.cat(position_ids, dim=0)


def _vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """Per-image cumulative patch boundaries."""
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    return F.pad(cu_seqlens, (1, 0), value=0)


class Qwen3_5VisionMLP(nn.Module):
    """Vision feed-forward: ``linear_fc2(gelu_tanh(linear_fc1(x)))``."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        hidden, inter = config.hidden_size, config.intermediate_size
        self.linear_fc1 = nn.Linear(hidden, inter)
        self.linear_fc2 = nn.Linear(inter, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class Qwen3_5VisionBlock(nn.Module):
    """Pre-norm ViT block: bidirectional attention + MLP.

    ``cu_seqlens`` marks per-image boundaries so attention stays within an image;
    ``position_embeddings`` is the ``(cos, sin)`` pair for the 2-D vision rotary.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.attn = VisionAttention(config.hidden_size, config.num_heads)
        self.mlp = Qwen3_5VisionMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens, position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3_5VisionModel(nn.Module):
    """Vision encoder: patch embed + bilinear pos-embed + ViT blocks + merger.

    Consumes ``pixel_values`` (pre-flattened patches) and a ``grid_thw`` tensor of
    per-image (temporal, height, width) patch counts, and returns patch-merged
    embeddings in the text model's hidden size. Follows the Hugging Face
    ``Qwen3VLVisionModel`` forward, reusing its position/interpolation utilities.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.patch_embed = VisionPatchEmbed(
            config.in_channels,
            config.hidden_size,
            config.patch_size,
            config.temporal_patch_size,
        )
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            Qwen3_5VisionBlock(config) for _ in range(config.depth)
        )
        self.merger = VisionPatchMerger(
            config.hidden_size, config.out_hidden_size, config.spatial_merge_size
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        # Bilinear-interpolate the learned pos-embed table to each image's grid.
        bilinear_indices, bilinear_weights = _vision_bilinear_indices_and_weights(
            grid_thw, self.num_grid_per_side, self.spatial_merge_size
        )
        position_ids = _vision_encoder_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = _vision_cu_seqlens(grid_thw)

        x = self.patch_embed(pixel_values)
        pos_embeds = (
            self.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]
        ).sum(0)
        x = x + pos_embeds.to(x.dtype)

        rotary = self.rotary_pos_emb(position_ids).reshape(x.shape[0], -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for block in self.blocks:
            x = block(x, cu_seqlens, position_embeddings)
        return self.merger(x)


class Qwen3_5Model(nn.Module):
    """Multimodal backbone: vision encoder + text decoder."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.image_token_id = config.image_token_id
        self.video_token_id = config.video_token_id
        self.spatial_merge_size = config.vision_config.spatial_merge_size
        self.visual = Qwen3_5VisionModel(config.vision_config)
        self.language_model = Qwen3_5TextModel(config.text_config)

    def get_vision_position_ids(
        self,
        start_position: Any,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """3-D (t, h, w) LLM position ids for one image/video, offset by
        ``start_position``."""
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size

        position_temporal = torch.arange(llm_grid_t, device=device) * time_interval
        position_height = torch.arange(llm_grid_h, device=device) + start_position
        position_width = torch.arange(llm_grid_w, device=device) + start_position

        t_grid, h_grid, w_grid = torch.meshgrid(
            position_temporal, position_height, position_width, indexing="ij"
        )
        vision_position_ids = torch.stack([t_grid, h_grid, w_grid], dim=0).reshape(
            3, -1
        )
        vision_position_ids[0] += start_position  # after time_interval multiply
        return vision_position_ids

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """3-D M-RoPE position ids ``(3, batch, seq)`` + per-sequence deltas.

        ``mm_token_type_ids`` marks each token text (0), image (1), or video (2).
        """
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                keep = attention_mask[batch_idx].bool()
                current_input_ids = current_input_ids[keep]
                input_token_type = input_token_type[keep]

            input_type_group = []
            for key, group in itertools.groupby(
                enumerate(input_token_type.tolist()), lambda x: x[1]
            ):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos: Any = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:  # text
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device)
                        .view(1, -1)
                        .expand(3, -1)
                        + current_pos
                    )
                    current_pos += text_len
                else:  # image (1) or video (2)
                    grid_iter = grid_iters[modality_type]
                    assert grid_iter is not None
                    grid_thw = next(grid_iter)
                    llm_pos_ids_list.append(
                        self.get_vision_position_ids(
                            current_pos,
                            grid_thw,
                            1,
                            spatial_merge_size,
                            device=input_ids.device,
                        )
                    )
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = (
                    llm_positions.to(position_ids.device)
                )
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(current_input_ids)
            )
        deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(
            1
        )
        return position_ids, deltas

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        *,
        cache: CacheView,
        batch: Batch | None = None,
    ) -> torch.Tensor:
        if pixel_values is None:
            return self.language_model(input_ids, positions, cache=cache, batch=batch)

        assert input_ids is not None and image_grid_thw is not None
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        image_embeds = self.visual(pixel_values, image_grid_thw)
        mask = input_ids == self.image_token_id
        inputs_embeds[mask] = image_embeds.to(inputs_embeds.dtype)

        if positions is None:
            # Reconstruct token types from the special ids for 3-D M-RoPE.
            if mm_token_type_ids is None:
                mm_token_type_ids = torch.zeros_like(input_ids)
                mm_token_type_ids[input_ids == self.image_token_id] = 1
                mm_token_type_ids[input_ids == self.video_token_id] = 2
            positions, _ = self.get_rope_index(
                input_ids, mm_token_type_ids, image_grid_thw
            )
        return self.language_model(
            positions=positions, inputs_embeds=inputs_embeds, cache=cache, batch=batch
        )


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Top-level Qwen3.5 model: backbone + LM head."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        text_config = config.text_config
        self.eos_token_id = text_config.eos_token_id
        self.model = Qwen3_5Model(config)
        self.sampler = Sampler()
        self.lm_head = nn.Linear(
            text_config.hidden_size, text_config.vocab_size, bias=False
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        *,
        cache: CacheView,
        batch: Batch | None = None,
    ) -> torch.Tensor:
        hidden = self.model(
            input_ids,
            positions,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
            cache=cache,
            batch=batch,
        )
        return self.lm_head(hidden)

    def make_cache(
        self, max_batch_size: int, max_seq_len: int, pages: int | None = None
    ) -> CachePool:
        """A decode-state pool: ``max_batch_size`` rows over ``pages`` of KV.

        What `walnut.scheduler.Scheduler` batches over; `iter_generate` builds
        its own single-row cache sized to the one request it serves.
        """
        return self.model.language_model.make_cache(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            dtype=self.lm_head.weight.dtype,
            device=self.lm_head.weight.device,
            pages=pages,
        )

    @torch.no_grad()
    def iter_generate(
        self,
        input_ids: torch.Tensor,
        params: SamplingParams | None = None,
        cuda_graph: bool = True,
        compile: bool = True,
        autotune: bool = True,
    ) -> Iterator[int]:
        """Yield generated token ids (text-only, batch 1), one per step.

        Sampling follows `params`. A stop id (any ``params.stop_token_ids``,
        defaulting to ``eos_token_id``) is yielded and then ends the stream.

        ``cuda_graph`` captures the decode step and replays it, trading a
        one-off capture for the per-step kernel launch cost (see `DecodeGraph`).
        It has no effect off CUDA.

        ``compile`` runs the decode step through `torch.compile`, which fuses
        the elementwise chains the norms and the delta-rule recurrence would
        otherwise spend a kernel apiece on. Prefill stays eager on purpose: its
        shapes follow the prompt, so compiling it recompiles per prompt length,
        while decode's are fixed and one compile serves every request.

        ``autotune`` lets Inductor benchmark a Triton template against cuBLAS
        for each projection instead of taking cuBLAS on faith. Decode's matmuls
        are matrix-*vector* products, a shape cuBLAS's `gemv` serves at 48-57%
        of this GPU's bandwidth where a Triton kernel reaches ~70%; picking per
        shape is worth ~11% of TPOT. Autotuning runs at compile time and its
        results land in the same on-disk cache as the compiled graph, so the
        cost is one cold compile per build, not one per process. Ignored
        without ``compile``.
        """
        params = params or SamplingParams()
        stop_ids = set(params.stop_token_ids)
        if not stop_ids and self.eos_token_id is not None:
            stop_ids = {self.eos_token_id}
        gen = None
        if params.seed is not None:
            gen = torch.Generator(device=input_ids.device).manual_seed(params.seed)

        seq = input_ids.shape[1]
        wanted = seq + params.max_new_tokens
        cache = self.model.language_model.make_cache(
            max_seq_len=wanted,
            max_batch_size=input_ids.shape[0],
            dtype=self.lm_head.weight.dtype,
            device=input_ids.device,
        )
        # Reserved, not merely allocated. A pool hands out pages through
        # `reserve`, and a row that never asked for any addresses `SCRATCH` for
        # every position it reaches — which reads as one page of cache silently
        # wrapping under the whole sequence, not as an error.
        for _ in range(input_ids.shape[0]):
            assert cache.reserve(wanted) is not None
        positions = torch.arange(seq, device=input_ids.device)
        # Decode positions carry a batch dim even at batch 1: the cache writes
        # and the attention mask read one position per row, and a fixed buffer
        # is what the captured graph copies into.
        decode_pos = torch.zeros(1, 1, dtype=torch.long, device=input_ids.device)

        # Separate functions so a profile can name the phases; inlining either
        # back into the loop leaves a trace that cannot be read per phase.
        # Both return the token twice, tensor and int: reading it back syncs on
        # the GPU, and that wait belongs to the step that caused it.

        def _prefill() -> tuple[torch.Tensor, int]:
            """Run the prompt through the model and sample the first token."""
            logits = self(input_ids, positions=positions, cache=cache)
            next_token = self.sampler(logits[:, -1], params, gen)
            return next_token, int(next_token.item())

        def _decode_step(
            token: torch.Tensor, position: int
        ) -> tuple[torch.Tensor, int]:
            """Advance one token: replay or forward, then sample."""
            decode_pos.fill_(position)
            if graph is not None:
                logits = graph.replay(token, decode_pos)
            else:
                logits = decode_forward(token, positions=decode_pos, cache=cache)
            next_token = self.sampler(logits[:, -1], params, gen)
            return next_token, int(next_token.item())

        next_token, tok = _prefill()

        # Hand the first token over before setting decode up. Everything below
        # this line serves the *second* token onwards, so holding the first one
        # behind it only adds its cost to TTFT; the caller gets the token as
        # soon as it exists and pays the setup while consuming it. A caller
        # that stops here never pays it at all.
        yield tok
        if tok in stop_ids:
            return

        # Compile after prefill, so the trace dynamo records is the decode
        # branch. Rebuilding the wrapper per request is a few milliseconds:
        # dynamo's own cache keys on the code object, not on this object.
        #
        # The autotuning mode is the "-no-cudagraphs" one because `DecodeGraph`
        # captures the step itself; letting Inductor also apply cudagraphs
        # would have it capture a region this code then captures again.
        decode_forward: Any = self
        if compile:
            mode = "max-autotune-no-cudagraphs" if autotune else None
            decode_forward = torch.compile(self, mode=mode)

        # Capture after prefill: the decode branch only exists once the caches
        # hold state, and capture records whichever branch it runs.
        graph = None
        if cuda_graph and input_ids.device.type == "cuda":
            graph = DecodeGraph(decode_forward, cache, input_ids.device)

        # The first token is already out, so this runs one step per *remaining*
        # token and samples nothing it will not yield.
        for step in range(params.max_new_tokens - 1):
            next_token, tok = _decode_step(next_token, seq + step)
            yield tok
            if tok in stop_ids:
                return

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        params: SamplingParams | None = None,
    ) -> torch.Tensor:
        """Return ``input_ids`` with generated tokens appended (batch 1)."""
        ids = list(self.iter_generate(input_ids, params))
        if not ids:
            return input_ids
        new = torch.tensor([ids], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, new], dim=1)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Copy checkpoint tensors into parameters by matching name.

        ``mtp`` is the multi-token prediction head, used for speculative
        decoding; walnut has no module for it, so those tensors are skipped.
        The model's `FusedLinear` layers each hold several of the checkpoint's
        projections; `copy_weights` finds them and concatenates their parts, so
        the checkpoint loads unmodified.
        """
        copy_weights(self, weights, skip_prefixes=("mtp.",))
