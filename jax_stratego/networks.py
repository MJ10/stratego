"""Flax NNX networks: MoveTransformer, ArrangementTransformer, BeliefTransformer, and belief masking."""

from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
from flax import nnx

from jax_stratego.env import (
    ACTION_TO_OCC_PAIR,
    ACTION_VALID_OCC,
    ARRANGEMENT_SIZE,
    FLAG,
    INFOSTATE_CHANNELS,
    N_OCCUPIABLE_CELL,
    N_PIECE_TYPE,
    OCCUPIABLE_INDICES,
)

N_VF_CAT = 3
CATEGORICAL_VALUES = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 128
    n_head: int = 4
    depth: int = 3
    ff_factor: int = 4
    use_cat_vf: bool = True


class SelfAttentionBlock(nnx.Module):
    def __init__(self, d_model: int, n_head: int, ff_factor: int, *, rngs: nnx.Rngs):
        self.n_head = n_head
        self.ln1 = nnx.LayerNorm(d_model, rngs=rngs)
        self.qkv = nnx.Linear(d_model, 3 * d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.ln2 = nnx.LayerNorm(d_model, rngs=rngs)
        self.ff1 = nnx.Linear(d_model, ff_factor * d_model, rngs=rngs)
        self.ff2 = nnx.Linear(ff_factor * d_model, d_model, rngs=rngs)

    def __call__(self, x: jax.Array, is_causal: bool = False) -> jax.Array:
        h = self.ln1(x)
        B, T, D = h.shape
        head_dim = D // self.n_head
        qkv = self.qkv(h).reshape(B, T, 3, self.n_head, head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = jax.nn.dot_product_attention(q, k, v, is_causal=is_causal)
        x = x + self.out_proj(attn.reshape(B, T, D))
        x = x + self.ff2(jax.nn.relu(self.ff1(self.ln2(x))))
        return x


class CrossAttentionBlock(nnx.Module):
    def __init__(self, d_model: int, n_head: int, ff_factor: int, *, rngs: nnx.Rngs):
        self.n_head = n_head
        self.self_block = SelfAttentionBlock(d_model, n_head, ff_factor, rngs=rngs)
        self.ln_q = nnx.LayerNorm(d_model, rngs=rngs)
        self.q_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.kv_proj = nnx.Linear(d_model, 2 * d_model, rngs=rngs)
        self.out_proj = nnx.Linear(d_model, d_model, rngs=rngs)
        self.ln_ff = nnx.LayerNorm(d_model, rngs=rngs)
        self.ff1 = nnx.Linear(d_model, ff_factor * d_model, rngs=rngs)
        self.ff2 = nnx.Linear(ff_factor * d_model, d_model, rngs=rngs)

    def __call__(self, x: jax.Array, mem: jax.Array) -> jax.Array:
        x = self.self_block(x, is_causal=True)
        B, T, D = x.shape
        S = mem.shape[1]
        head_dim = D // self.n_head
        q = self.q_proj(self.ln_q(x)).reshape(B, T, self.n_head, head_dim)
        kv = self.kv_proj(mem).reshape(B, S, 2, self.n_head, head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]
        attn = jax.nn.dot_product_attention(q, k, v, is_causal=False)
        x = x + self.out_proj(attn.reshape(B, T, D))
        x = x + self.ff2(jax.nn.relu(self.ff1(self.ln_ff(x))))
        return x


class MoveTransformer(nnx.Module):
    """92-cell board Transformer with QK^T src-dst action head and categorical value head."""

    def __init__(self, cfg: TransformerConfig = TransformerConfig(), *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embedder = nnx.Linear(INFOSTATE_CHANNELS, cfg.d_model, rngs=rngs)
        self.pos_emb = nnx.Param(
            jax.random.normal(rngs.params(), (1, N_OCCUPIABLE_CELL + 1, cfg.d_model)) * 0.02
        )
        self.blocks = nnx.List(
            [SelfAttentionBlock(cfg.d_model, cfg.n_head, cfg.ff_factor, rngs=rngs) for _ in range(cfg.depth)]
        )
        self.norm_out = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.q_proj = nnx.Linear(cfg.d_model, cfg.d_model, rngs=rngs)
        self.k_proj = nnx.Linear(cfg.d_model, cfg.d_model, rngs=rngs)
        self.value_head = nnx.Linear(
            cfg.d_model, N_VF_CAT if cfg.use_cat_vf else 1, rngs=rngs
        )

    def __call__(
        self, infostate: jax.Array, legal_action_mask: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Forward pass.

        Args:
            infostate: (B, 92, C)
            legal_action_mask: (B, 1800) bool

        Returns:
            action_log_probs: (B, 1800)
            value_out: (B, 3) log-probs if use_cat_vf else (B,) scalar
        """
        B = infostate.shape[0]
        x = self.embedder(infostate)
        # Prepend 1 special value token
        val_tok = jnp.zeros((B, 1, self.cfg.d_model), dtype=x.dtype)
        x = jnp.concatenate([val_tok, x], axis=1) + self.pos_emb[...]
        for block in self.blocks:
            x = block(x, is_causal=False)
        out = self.norm_out(x)

        cell_emb = out[:, 1:, :]  # (B, 92, D)
        q = self.q_proj(cell_emb)
        k = self.k_proj(cell_emb)
        pair_scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(self.cfg.d_model)  # (B, 92, 92)
        flat_scores = pair_scores.reshape(B, N_OCCUPIABLE_CELL * N_OCCUPIABLE_CELL)

        # Map (src_occ, dst_occ) -> 1800 actions (matches LogitConverter)
        raw_logits = flat_scores[:, ACTION_TO_OCC_PAIR]
        masked_logits = jnp.where(legal_action_mask & ACTION_VALID_OCC[None, :], raw_logits, -1e9)
        action_log_probs = jax.nn.log_softmax(masked_logits, axis=-1)

        v = self.value_head(out[:, 0, :])
        if self.cfg.use_cat_vf:
            v = jax.nn.log_softmax(v, axis=-1)
        else:
            v = jnp.squeeze(v, axis=-1)
        return action_log_probs, v


class ArrangementTransformer(nnx.Module):
    """Autoregressive Transformer generating initial 40-cell piece setups."""

    def __init__(
        self,
        piece_counts: jax.Array,
        cfg: TransformerConfig = TransformerConfig(d_model=128, n_head=4, depth=2),
        *,
        rngs: nnx.Rngs,
    ):
        self.cfg = cfg
        self.piece_counts = piece_counts  # int32[14], sum == 40
        self.embedder = nnx.Linear(N_PIECE_TYPE, cfg.d_model, rngs=rngs)
        self.pos_emb = nnx.Param(
            jax.random.normal(rngs.params(), (1, ARRANGEMENT_SIZE, cfg.d_model)) * 0.02
        )
        self.blocks = nnx.List(
            [SelfAttentionBlock(cfg.d_model, cfg.n_head, cfg.ff_factor, rngs=rngs) for _ in range(cfg.depth)]
        )
        self.norm_out = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.policy_out = nnx.Linear(cfg.d_model, N_PIECE_TYPE, rngs=rngs)
        self.value_out = nnx.Linear(cfg.d_model, N_VF_CAT if cfg.use_cat_vf else 1, rngs=rngs)

    def _legal_mask(self, seq_onehot: jax.Array) -> jax.Array:
        """Compute legal piece-type mask (B, 40, 14) from shifted prefix seq_onehot (B, 40, 14)."""
        cum_counts = jnp.cumsum(seq_onehot.astype(jnp.int32), axis=1)
        remaining = self.piece_counts[None, None, :] - cum_counts
        legal = remaining > 0
        # Enforce right-side flag placement (cols 5..9 in each of the 4 rows) to break left-right symmetry
        right_side = (jnp.arange(ARRANGEMENT_SIZE) % 10) >= 5  # bool[40]
        flag_allowed = right_side[None, :]
        legal = legal.at[:, :, FLAG].set(legal[:, :, FLAG] & flag_allowed)
        return legal

    def __call__(self, arrangements_onehot: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Teacher-forced forward pass on arrangements_onehot: (B, 40, 14)."""
        B = arrangements_onehot.shape[0]
        start_tok = jnp.zeros((B, 1, N_PIECE_TYPE), dtype=arrangements_onehot.dtype)
        seq = jnp.concatenate([start_tok, arrangements_onehot[:, :-1, :]], axis=1)

        x = self.embedder(seq) + self.pos_emb[...]
        for block in self.blocks:
            x = block(x, is_causal=True)
        x = self.norm_out(x)

        logits = self.policy_out(x)
        legal_mask = self._legal_mask(seq)
        logits = jnp.where(legal_mask, logits, -1e9)
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        v = self.value_out(x)
        if self.cfg.use_cat_vf:
            v = jax.nn.log_softmax(v, axis=-1)
        else:
            v = jnp.squeeze(v, axis=-1)
        return log_probs, v


def sample_arrangements(
    model: ArrangementTransformer, key: jax.Array, n_sample: int, randomize_flip: bool = True
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Autoregressively sample `n_sample` valid arrangements using `jax.lax.scan`.

    Returns:
        arr_ids: int8[n_sample, 40] (oriented for the environment)
        canonical_onehot: float32[n_sample, 40, 14] (before horizontal flip, for PPO training)
        log_probs: float32[n_sample, 40, 14]
        values: float32[n_sample, 40, 3]
    """
    init_samples = jnp.zeros((n_sample, ARRANGEMENT_SIZE, N_PIECE_TYPE), dtype=jnp.float32)
    step_keys = jax.random.split(key, ARRANGEMENT_SIZE + 1)
    flip_key, step_keys = step_keys[0], step_keys[1:]
    graphdef, state = nnx.split(model)

    def scan_step(samples, idx_and_key):
        t, k = idx_and_key
        m = nnx.merge(graphdef, state)
        log_pi, v = m(samples)
        log_pi_t = log_pi[:, t, :]
        v_t = v[:, t]
        sampled_piece = jax.random.categorical(k, log_pi_t, axis=-1)
        onehot_t = jax.nn.one_hot(sampled_piece, N_PIECE_TYPE, dtype=jnp.float32)
        samples = samples.at[:, t, :].set(onehot_t)
        return samples, (log_pi_t, v_t)

    canonical_onehot, (log_probs_T, values_T) = jax.lax.scan(
        scan_step, init_samples, (jnp.arange(ARRANGEMENT_SIZE), step_keys)
    )
    log_probs = jnp.swapaxes(log_probs_T, 0, 1)  # (B, 40, 14)
    values = jnp.swapaxes(values_T, 0, 1)        # (B, 40, 3)

    arr_ids = jnp.argmax(canonical_onehot, axis=-1).astype(jnp.int8)  # (B, 40)
    if randomize_flip:
        do_flip = jax.random.bernoulli(flip_key, 0.5, shape=(n_sample,))
        flipped_ids = jnp.flip(arr_ids.reshape(n_sample, 4, 10), axis=-1).reshape(n_sample, 40)
        arr_ids = jnp.where(do_flip[:, None], flipped_ids, arr_ids)

    return arr_ids, canonical_onehot, log_probs, values


def create_belief_mask(
    unknown_piece_counts: jax.Array,
    unknown_piece_type_onehot: jax.Array,
    unknown_piece_has_moved: jax.Array,
) -> jax.Array:
    """Create feasible piece-type mask (B, 40, 14) for unknown opponent pieces.

    Matches `pyengine/belief/masking.py`:
    1. Piece counts cannot exceed `unknown_piece_counts` (B, 12).
    2. Moved pieces cannot be immovable (Flag=10, Bomb=11).
    3. Unmoved pieces cannot be assigned a movable type if remaining unmoved slots == remaining immovable pieces.
    """
    B, n_piece, _ = unknown_piece_type_onehot.shape

    # 1. Piece count mask
    piece_counts_14 = jnp.pad(unknown_piece_counts, ((0, 0), (0, 2)))  # (B, 14)
    oh_int = unknown_piece_type_onehot.astype(jnp.int32)
    cum_counts = jnp.cumsum(oh_int, axis=1) - oh_int
    count_mask = cum_counts < piece_counts_14[:, None, :]

    total_counts = jnp.sum(piece_counts_14, axis=1, keepdims=True)  # (B, 1)
    idx_range = jnp.arange(n_piece)[None, :]
    is_padding = idx_range >= total_counts
    count_mask = count_mask | is_padding[:, :, None]

    # 2. Movement mask
    # Disallow Flag(10) and Bomb(11) on moved pieces
    not_moved = ~unknown_piece_has_moved  # (B, 40)
    immovable_assigned = oh_int[:, :, 10:12]
    total_immovable = jnp.sum(unknown_piece_counts[:, 10:12], axis=1, keepdims=True)  # (B, 1)
    # Exclusive prefix sum of immovable assignments before slot i
    cum_immovable_before = jnp.sum(jnp.cumsum(immovable_assigned, axis=1) - immovable_assigned, axis=-1)
    rem_immovable = total_immovable - cum_immovable_before  # (B, 40)

    unmoved_active = not_moved & (~is_padding)
    rem_unmoved = jnp.flip(jnp.cumsum(jnp.flip(unmoved_active.astype(jnp.int32), axis=1), axis=1), axis=1)

    # If remaining unmoved slots == remaining immovable pieces, unmoved slot MUST take an immovable piece
    must_be_immovable = (rem_immovable >= rem_unmoved) & unmoved_active
    allow_movable = ~must_be_immovable  # (B, 40)

    type_idx = jnp.arange(N_PIECE_TYPE)[None, None, :]
    is_movable_type = type_idx < 10
    is_immovable_type = (type_idx == 10) | (type_idx == 11)
    move_mask = (
        (is_movable_type & allow_movable[:, :, None])
        | (is_immovable_type & not_moved[:, :, None])
        | is_padding[:, :, None]
    )

    return count_mask & move_mask


class BeliefTransformer(nnx.Module):
    """Encoder-Decoder Transformer predicting joint distribution over unknown opponent pieces."""

    def __init__(
        self,
        cfg: TransformerConfig = TransformerConfig(d_model=128, n_head=4, depth=2),
        *,
        rngs: nnx.Rngs,
    ):
        self.cfg = cfg
        self.embedder = nnx.Linear(INFOSTATE_CHANNELS, cfg.d_model, rngs=rngs)
        self.pos_emb_enc = nnx.Param(
            jax.random.normal(rngs.params(), (1, N_OCCUPIABLE_CELL, cfg.d_model)) * 0.02
        )
        self.piece_embed = nnx.Linear(N_PIECE_TYPE, cfg.d_model, rngs=rngs)
        self.pos_emb_dec = nnx.Param(
            jax.random.normal(rngs.params(), (1, ARRANGEMENT_SIZE, cfg.d_model)) * 0.02
        )
        self.encoder = nnx.List(
            [SelfAttentionBlock(cfg.d_model, cfg.n_head, cfg.ff_factor, rngs=rngs) for _ in range(cfg.depth)]
        )
        self.decoder = nnx.List(
            [CrossAttentionBlock(cfg.d_model, cfg.n_head, cfg.ff_factor, rngs=rngs) for _ in range(cfg.depth)]
        )
        self.ln_enc = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.ln_dec = nnx.LayerNorm(cfg.d_model, rngs=rngs)
        self.out_proj = nnx.Linear(cfg.d_model, N_PIECE_TYPE, rngs=rngs)

    def encode(self, infostate: jax.Array, unknown_pos_onehot: jax.Array) -> jax.Array:
        """Encode board infostate (B, 92, C) and gather embeddings for unknown piece positions (B, 40, 100)."""
        x = self.embedder(infostate) + self.pos_emb_enc[...]
        for block in self.encoder:
            x = block(x, is_causal=False)
        # Extract embeddings at occupiable cells corresponding to each unknown piece
        occ_pos = unknown_pos_onehot[:, :, OCCUPIABLE_INDICES]  # (B, 40, 92)
        occ_idx = jnp.argmax(occ_pos.astype(jnp.int32), axis=-1)  # (B, 40)
        is_active = jnp.any(occ_pos, axis=-1, keepdims=True)  # (B, 40, 1)
        gathered = jnp.take_along_axis(x, occ_idx[:, :, None], axis=1)
        mem = jnp.where(is_active, gathered, 0.0)
        return self.ln_enc(mem)

    def decode(self, unknown_type_onehot: jax.Array, mem: jax.Array) -> jax.Array:
        """Decode shifted unknown piece types (B, 40, 14) conditioned on encoder memory (B, 40, D)."""
        B = unknown_type_onehot.shape[0]
        start_tok = jnp.zeros((B, 1, N_PIECE_TYPE), dtype=jnp.float32)
        shifted = jnp.concatenate([start_tok, unknown_type_onehot[:, :-1, :].astype(jnp.float32)], axis=1)
        x = self.piece_embed(shifted) + self.pos_emb_dec[...]
        for block in self.decoder:
            x = block(x, mem)
        return self.out_proj(self.ln_dec(x))

    def __call__(
        self,
        infostate: jax.Array,
        unknown_pos_onehot: jax.Array,
        unknown_type_onehot: jax.Array,
        unknown_counts: jax.Array,
        unknown_has_moved: jax.Array,
    ) -> jax.Array:
        """Return masked log-probabilities (B, 40, 14) over unknown piece types."""
        mem = self.encode(infostate, unknown_pos_onehot)
        logits = self.decode(unknown_type_onehot, mem)
        mask = create_belief_mask(unknown_counts, unknown_type_onehot, unknown_has_moved)
        masked_logits = jnp.where(mask, logits, -1e9)
        return jax.nn.log_softmax(masked_logits, axis=-1)


def sample_belief(
    model: BeliefTransformer,
    key: jax.Array,
    n_sample: int,
    infostate: jax.Array,
    unknown_pos_onehot: jax.Array,
    unknown_has_moved: jax.Array,
    unknown_counts: jax.Array,
    max_unknown: int = ARRANGEMENT_SIZE,
) -> jax.Array:
    """Autoregressively sample `n_sample` consistent hidden piece assignments (n_sample, 40, 14)."""
    mem_1 = model.encode(infostate[None, ...], unknown_pos_onehot[None, ...])
    mem = jnp.repeat(mem_1, n_sample, axis=0)  # (n_sample, 40, D)

    has_moved_B = jnp.repeat(unknown_has_moved[None, :], n_sample, axis=0)
    counts_B = jnp.repeat(unknown_counts[None, :], n_sample, axis=0)
    active_slots = jnp.any(unknown_pos_onehot, axis=-1)  # (40,)

    init_samples = jnp.zeros((n_sample, ARRANGEMENT_SIZE, N_PIECE_TYPE), dtype=jnp.bool_)
    step_keys = jax.random.split(key, max_unknown)
    graphdef, state = nnx.split(model)

    def scan_step(samples, idx_and_key):
        i, k = idx_and_key
        m = nnx.merge(graphdef, state)
        logits = m.decode(samples, mem)
        mask = create_belief_mask(counts_B, samples, has_moved_B)
        masked_logits_i = jnp.where(mask[:, i, :], logits[:, i, :], -1e9)
        sampled_piece = jax.random.categorical(k, masked_logits_i, axis=-1)
        onehot_i = jax.nn.one_hot(sampled_piece, N_PIECE_TYPE, dtype=jnp.bool_)
        onehot_i = jnp.where(active_slots[i], onehot_i, False)
        samples = samples.at[:, i, :].set(onehot_i)
        return samples, None

    samples, _ = jax.lax.scan(scan_step, init_samples, (jnp.arange(max_unknown), step_keys))
    return samples
