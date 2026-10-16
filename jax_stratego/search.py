"""Vectorized test-time SearchBot in pure JAX + Flax NNX (matching `pyengine/core/search.py`)."""

from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
from flax import nnx

from jax_stratego.env import (
    EnvState,
    N_ACTION,
    assign_opponent_hidden_pieces,
    get_unknown_piece_info,
    legal_action_mask,
    observe,
    step_env,
)
from jax_stratego.networks import (
    BeliefTransformer,
    CATEGORICAL_VALUES,
    MoveTransformer,
    N_VF_CAT,
    sample_belief,
)
from jax_stratego.train import get_weighted_uniform_policy


@dataclass(frozen=True)
class SearchConfig:
    num_rollout_envs: int = 128
    depth: int = 4  # Must be even (full player-opponent ply pairs)
    stepsize: float = 5.0
    temperature: float = 0.05
    td_lambda: float = 0.8
    max_num_Moves: int = 400
    max_moves_between_attacks: int = 80
    max_unknown: int = 8
    uniform_magnet: bool = False


def allocate_rollout_actions(mask: jax.Array, num_rollout_envs: int) -> jax.Array:
    """Deterministically allocate `num_rollout_envs` samples evenly across legal root actions.

    Static-shape equivalent of `sample_deterministic` in `pyengine/core/search.py`.
    """
    n_legal = jnp.maximum(jnp.sum(mask.astype(jnp.int32)), 1)
    # Rank legal actions 0 .. n_legal-1
    legal_ranks = jnp.cumsum(mask.astype(jnp.int32)) - 1
    target_ranks = jnp.arange(num_rollout_envs, dtype=jnp.int32) % n_legal
    # Map each target rank back to its action index in 0..1799
    matches = mask[:, None] & (legal_ranks[:, None] == target_ranks[None, :])
    return jnp.argmax(matches.astype(jnp.int32), axis=0)  # (num_rollout_envs,)


def compute_search_policy(
    q_estimates: jax.Array,       # (1800,)
    bp_log_probs: jax.Array,      # (1800,)
    legal_mask: jax.Array,        # (1800,) bool
    temperature: float,
    stepsize: float,
    uniform_magnet: bool = False,
) -> jax.Array:
    """Closed-form KL/magnet-regularized mirror descent policy update (`pyengine/core/search.py`)."""
    temp_ss = temperature * stepsize
    if uniform_magnet:
        search_logits = (bp_log_probs + stepsize * q_estimates) / (1.0 + temp_ss)
    else:
        magnet = get_weighted_uniform_policy(legal_mask[None, :])[0]
        log_magnet = jnp.log(jnp.maximum(magnet, 1e-10))
        search_logits = (bp_log_probs + stepsize * q_estimates + temp_ss * log_magnet) / (
            1.0 + temp_ss
        )
    masked_logits = jnp.where(legal_mask, search_logits, -1e9)
    return jax.nn.softmax(masked_logits, axis=-1)


def estimate_q_values(
    move_model: MoveTransformer,
    rollout_states: EnvState,     # batched EnvState of size R = num_rollout_envs
    root_legal_mask: jax.Array,   # (1800,) bool
    key: jax.Array,
    cfg: SearchConfig,
) -> tuple[jax.Array, jax.Array]:
    """Run depth-limited two-player rollouts and compute TD(lambda) Q-estimates for root actions.

    Matches `SearchBot.estimate_q_values` in `pyengine/core/search.py`.
    """
    R = cfg.num_rollout_envs
    root_player = rollout_states.to_play[0]  # scalar (0 or 1)
    root_sign = jnp.where(root_player == 0, 1.0, -1.0)

    root_actions = allocate_rollout_actions(root_legal_mask, R)  # (R,)

    vmap_step = jax.vmap(
        lambda s, a: step_env(s, a, cfg.max_num_Moves, cfg.max_moves_between_attacks)
    )
    vmap_obs = jax.vmap(
        lambda s: observe(s, cfg.max_num_Moves, cfg.max_moves_between_attacks)
    )
    vmap_mask = jax.vmap(legal_action_mask)

    move_graphdef, move_state = nnx.split(move_model)
    num_pairs = cfg.depth // 2
    pair_keys = jax.random.split(key, num_pairs)

    def depth_pair_body(carry, d_and_key):
        env_s, act, leaf_vals, done = carry
        d, pair_key = d_and_key
        k0, k1 = jax.random.split(pair_key)
        model = nnx.merge(move_graphdef, move_state)

        # Parity 0: root player's move (at d=0 this applies `root_actions`)
        s_after_0 = vmap_step(env_s, act)
        newly_term_0 = s_after_0.done & (~env_s.done)
        rew_0 = newly_term_0.astype(jnp.float32) * s_after_0.reward_pl0

        obs_1 = vmap_obs(s_after_0)
        mask_1 = vmap_mask(s_after_0)
        logp_1, _ = model(obs_1, mask_1)
        act_1 = jax.random.categorical(k0, logp_1, axis=-1)

        # Parity 1: opponent's move, bringing turn back to root_player
        s_after_1 = vmap_step(s_after_0, act_1)
        newly_term_1 = s_after_1.done & (~s_after_0.done)
        rew_total = (rew_0 + newly_term_1.astype(jnp.float32) * s_after_1.reward_pl0) * root_sign

        obs_2 = vmap_obs(s_after_1)
        mask_2 = vmap_mask(s_after_1)
        logp_2, val_logp_2 = model(obs_2, mask_2)
        next_act = jax.random.categorical(k1, logp_2, axis=-1)

        term = s_after_1.done
        rew_cat = jax.nn.one_hot(jnp.round(rew_total).astype(jnp.int32) + 1, N_VF_CAT, dtype=jnp.float32)
        val_probs = jax.nn.softmax(val_logp_2, axis=-1)

        is_last = d == (num_pairs - 1)
        lam_d = jnp.float32(cfg.td_lambda) ** d.astype(jnp.float32)
        normalizer = jnp.where(is_last, lam_d, (1.0 - jnp.float32(cfg.td_lambda)) * lam_d)
        target = jnp.where(term[:, None], lam_d * rew_cat, normalizer * val_probs)

        leaf_vals = leaf_vals + (~done)[:, None].astype(jnp.float32) * target
        done = done | term
        return (s_after_1, next_act, leaf_vals, done), None

    init_carry = (
        rollout_states,
        root_actions,
        jnp.zeros((R, N_VF_CAT), dtype=jnp.float32),
        rollout_states.done,
    )
    (_, _, leaf_vals, _), _ = jax.lax.scan(
        depth_pair_body, init_carry, (jnp.arange(num_pairs), pair_keys)
    )

    # Scatter-average leaf categorical values per root action
    cum_q = jnp.zeros((N_ACTION, N_VF_CAT), dtype=jnp.float32).at[root_actions].add(leaf_vals)
    counts = jnp.zeros(N_ACTION, dtype=jnp.float32).at[root_actions].add(1.0)
    cat_q = cum_q / jnp.maximum(counts[:, None], 1.0)
    scalar_q = jnp.matmul(cat_q, CATEGORICAL_VALUES)  # (1800,)
    return scalar_q, counts


def search_action(
    move_model: MoveTransformer,
    belief_model: Optional[BeliefTransformer],
    state: EnvState,              # single unbatched EnvState
    key: jax.Array,
    cfg: SearchConfig = SearchConfig(),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute test-time search policy and sample an action for a single `EnvState`.

    1. Determinizes hidden opponent pieces across `cfg.num_rollout_envs` worlds using `belief_model`
       (or uses ground-truth state if `belief_model` is None).
    2. Runs depth-`cfg.depth` rollouts with TD(lambda) value backups.
    3. Combines Q-estimates with root policy logits via regularized mirror descent.

    Returns:
        action: int32 scalar
        search_policy: float32[1800]
        q_estimates: float32[1800]
    """
    k_bel, k_roll, k_act = jax.random.split(key, 3)
    R = cfg.num_rollout_envs

    obs = observe(state, cfg.max_num_Moves, cfg.max_moves_between_attacks)
    mask = legal_action_mask(state)

    # Tile state to R parallel rollout environments
    tiled_state = jax.tree.map(
        lambda x: jnp.repeat(x[None, ...], R, axis=0), state
    )

    if belief_model is not None:
        pos_oh, _, moved, counts = get_unknown_piece_info(state)
        sampled_hidden = sample_belief(
            belief_model, k_bel, R, obs, pos_oh, moved, counts, max_unknown=cfg.max_unknown
        )
        rollout_states = jax.vmap(assign_opponent_hidden_pieces)(tiled_state, sampled_hidden)
    else:
        rollout_states = tiled_state

    q_estimates, _ = estimate_q_values(move_model, rollout_states, mask, k_roll, cfg)
    bp_log_probs, _ = move_model(obs[None, ...], mask[None, ...])
    bp_log_probs = bp_log_probs[0]

    search_policy = compute_search_policy(
        q_estimates, bp_log_probs, mask, cfg.temperature, cfg.stepsize, cfg.uniform_magnet
    )
    action = jax.random.categorical(k_act, jnp.log(jnp.maximum(search_policy, 1e-12)))
    return action, search_policy, q_estimates
