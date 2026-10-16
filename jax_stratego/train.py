"""End-to-end JIT-compiled RL (Move PPO + Arrangement PPO + Belief CE) training in JAX & Flax NNX."""

from dataclasses import dataclass
from typing import NamedTuple
import jax
import jax.numpy as jnp
from flax import nnx
import optax

from jax_stratego.env import (
    ACTION_SRC,
    ARRANGEMENT_SIZE,
    BARRAGE_COUNTS,
    CLASSIC_COUNTS,
    EnvState,
    N_BOARD_CELL,
    N_PIECE_TYPE,
    get_unknown_piece_info,
    init_env,
    legal_action_mask,
    observe,
    step_env,
)
from jax_stratego.networks import (
    ArrangementTransformer,
    BeliefTransformer,
    CATEGORICAL_VALUES,
    MoveTransformer,
    N_VF_CAT,
    TransformerConfig,
    sample_arrangements,
)


@dataclass(frozen=True)
class TrainConfig:
    barrage: bool = True
    num_envs: int = 32
    rollout_steps: int = 64  # Must be even (alternating players)
    max_num_moves: int = 400
    max_moves_between_attacks: int = 80
    # RL / R-NaD-style regularization & schedule hyperparameters (matching `pyengine/core/rl.py`)
    lr: float = 3e-4
    lr_coef: float = 0.5
    lr_decay: float = 1.1
    lr_floor: float = 5e-6
    use_power_schedule: bool = False
    clip_range: float = 0.2
    td_lambda: float = 0.8
    gae_lambda: float = 0.5
    vf_coef: float = 1.0
    policy_coef: float = 1.0
    kl_coef: float = 0.1
    temperature: float = 0.02
    temperature_coef: float = 0.05
    temperature_ceil: float = 0.1
    temperature_floor: float = 0.001
    temperature_decay: float = 0.3
    adv_filt_rate: float = 0.75
    adv_filt_thresh: float = 0.01
    uniform_magnet: bool = True
    ema_decay: float = 0.995
    max_grad_norm: float = 0.5
    # Arrangement RL hyperparameters (matching `ArrangementBuffer.process_data` & `RL.arr_train`)
    arr_lr: float = 5e-5
    arr_td_lambda: float = 1.0
    arr_gae_lambda: float = 1.0
    arr_vf_coef: float = 0.5
    arr_ent_pred_coef: float = 1.0
    arr_reg_norm: float = 10.0
    arr_temperature: float = 0.05
    arr_temperature_coef: float = 0.1
    arr_temperature_ceil: float = 1.0
    arr_temperature_floor: float = 0.001
    arr_temperature_decay: float = 0.3
    # Model dimensions
    d_model: int = 64
    n_head: int = 4
    depth: int = 2
    use_bfloat16: bool = False


class Transition(NamedTuple):
    infostate: jax.Array            # (B, 92, C)
    legal_mask: jax.Array           # (B, 1800)
    action: jax.Array               # (B,)
    log_probs: jax.Array            # (B, 1800)
    value_log_probs: jax.Array      # (B, 3)
    acting_reward: jax.Array        # (B,) reward from acting player's POV
    newly_done: jax.Array           # (B,) bool
    unknown_pos_onehot: jax.Array   # (B, 40, 100)
    unknown_type_onehot: jax.Array  # (B, 40, 14)
    unknown_has_moved: jax.Array    # (B, 40)
    unknown_counts: jax.Array       # (B, 12)


class ArrEpisodeBatch(NamedTuple):
    canonical_onehot: jax.Array     # (B, 2, 40, 14)
    old_log_probs: jax.Array        # (B, 2, 40, 14)
    old_values: jax.Array           # (B, 2, 40, 3) log-probs
    old_ent_preds: jax.Array        # (B, 2, 40) normalized future entropy predictions


def power_schedule(
    coef: float, step: jax.Array, decay: float, ceil: float, floor: float
) -> jax.Array:
    """Power-law schedule `clip(coef / (step + 1)^decay, floor, ceil)` (`pyengine/core/rl.py:756`)."""
    val = jnp.float32(coef) / ((step.astype(jnp.float32) + 1.0) ** jnp.float32(decay))
    return jnp.clip(val, jnp.float32(floor), jnp.float32(ceil))


def get_weighted_uniform_policy(legal_action_mask: jax.Array) -> jax.Array:
    """Piece-uniform then action-uniform magnet distribution (`pyengine/utils/helper.py:20`).

    Args:
        legal_action_mask: (N, 1800) bool
    Returns:
        magnet_probs: (N, 1800) float32
    """
    # Count legal actions originating from each board cell (0..99)
    mask_i32 = legal_action_mask.astype(jnp.int32)
    cell_one_hot = jax.nn.one_hot(ACTION_SRC, N_BOARD_CELL, dtype=jnp.int32)  # (1800, 100)
    cell_counts = jnp.matmul(mask_i32, cell_one_hot)  # (N, 100)
    expanded_counts = jnp.maximum(cell_counts[:, ACTION_SRC], 1)  # (N, 1800)
    unnorm = legal_action_mask.astype(jnp.float32) / expanded_counts.astype(jnp.float32)
    return unnorm / jnp.maximum(jnp.sum(unnorm, axis=-1, keepdims=True), 1e-8)


def compute_two_player_gae(
    value_log_probs: jax.Array,  # (T, B, 3)
    acting_reward: jax.Array,    # (T, B)
    newly_done: jax.Array,       # (T, B) bool
    bootstrap_val_p0: jax.Array, # (B, 3) log-probs at T
    bootstrap_val_p1: jax.Array, # (B, 3) log-probs at T+1
    td_lambda: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute solipsistic 2-player alternating TD(lambda) categorical returns and scalar GAE advantages."""
    next_step_done = jnp.concatenate([newly_done[1:], jnp.zeros_like(newly_done[:1])], axis=0)
    next_step_rew = jnp.concatenate([acting_reward[1:], jnp.zeros_like(acting_reward[:1])], axis=0)

    is_term = newly_done | next_step_done  # (T, B)
    term_rew = jnp.where(newly_done, acting_reward, jnp.where(next_step_done, -next_step_rew, 0.0))
    term_cat = jax.nn.one_hot(jnp.round(term_rew).astype(jnp.int32) + 1, N_VF_CAT, dtype=jnp.float32)

    val_probs = jax.nn.softmax(value_log_probs, axis=-1)  # (T, B, 3)
    next_val_probs = jnp.concatenate(
        [
            val_probs[2:],
            jax.nn.softmax(bootstrap_val_p0[None, ...], axis=-1),
            jax.nn.softmax(bootstrap_val_p1[None, ...], axis=-1),
        ],
        axis=0,
    )

    target_val_probs = jnp.where(is_term[:, :, None], term_cat, next_val_probs)

    half_T = value_log_probs.shape[0] // 2
    B = value_log_probs.shape[1]
    vp_2 = val_probs.reshape(half_T, 2, B, N_VF_CAT)
    tv_2 = target_val_probs.reshape(half_T, 2, B, N_VF_CAT)
    term_2 = is_term.reshape(half_T, 2, B)

    scalar_v = jnp.matmul(vp_2, CATEGORICAL_VALUES)
    scalar_tv = jnp.matmul(tv_2, CATEGORICAL_VALUES)
    deltas = scalar_tv - scalar_v

    def scan_fn(carry, inputs):
        ret_next, gae_next = carry
        tv_t, delta_t, done_t = inputs
        ret_t = tv_t + td_lambda * (~done_t)[:, :, None] * (ret_next - tv_t)
        gae_t = delta_t + gae_lambda * (~done_t) * gae_next
        return (ret_t, gae_t), (ret_t, gae_t)

    init_carry = (tv_2[-1], jnp.zeros_like(deltas[-1]))
    _, (returns_2, adv_2) = jax.lax.scan(
        scan_fn, init_carry, (tv_2, deltas, term_2), reverse=True
    )
    return returns_2.reshape(-1, B, N_VF_CAT), adv_2.reshape(-1, B)


def compute_arrangement_prefix_gae(
    canonical_onehot: jax.Array,  # (N, 40, 14)
    old_log_probs: jax.Array,     # (N, 40, 14)
    old_val_log_probs: jax.Array, # (N, 40, 3)
    old_ent_preds: jax.Array,     # (N, 40)
    terminal_cat_ret: jax.Array,  # (N, 3) categorical terminal game return
    td_lambda: float,
    gae_lambda: float,
    reg_temp: jax.Array,
    reg_norm: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute 40-step prefix TD(lambda) returns, entropy returns, and GAE advantages.

    Matches `ArrangementBuffer.process_data` in `pyengine/arrangement/buffer.py:293-360`:
    1. Grounded game-outcome TD(lambda) & GAE across prefixes k = 39..0.
    2. Future accumulated entropy TD(lambda) & GAE across prefixes k = 39..0 using step NLL `-log_prob[k]`.
    3. Total advantage = grounded_adv + reg_temp * entropy_adv.
    """
    val_probs = jax.nn.softmax(old_val_log_probs, axis=-1)  # (N, 40, 3)
    ents = reg_norm * old_ent_preds                         # (N, 40)

    chosen_idx = jnp.argmax(canonical_onehot, axis=-1, keepdims=True)
    nll = -jnp.take_along_axis(old_log_probs, chosen_idx, axis=-1).squeeze(-1)  # (N, 40)

    # Transpose to (40, N, ...) for reverse scan over placement steps 39..0
    vp_T = jnp.swapaxes(val_probs, 0, 1)  # (40, N, 3)
    ents_T = jnp.swapaxes(ents, 0, 1)     # (40, N)
    nll_T = jnp.swapaxes(nll, 0, 1)       # (40, N)

    next_vp_T = jnp.concatenate([vp_T[1:], terminal_cat_ret[None, :, :]], axis=0)  # (40, N, 3)
    next_ents_T = jnp.concatenate([ents_T[1:], jnp.zeros_like(ents_T[:1])], axis=0)  # (40, N)

    val_deltas_T = next_vp_T - vp_T                        # (40, N, 3)
    reg_deltas_T = nll_T + next_ents_T - ents_T            # (40, N)

    def scan_fn(carry, inputs):
        td_tr, gae_tr, reg_td_tr, reg_gae_tr = carry
        v_delta, r_delta = inputs
        td_tr = v_delta + td_lambda * td_tr
        gae_tr = v_delta + gae_lambda * gae_tr
        reg_td_tr = r_delta + td_lambda * reg_td_tr
        reg_gae_tr = r_delta + gae_lambda * reg_gae_tr
        return (td_tr, gae_tr, reg_td_tr, reg_gae_tr), (td_tr, gae_tr, reg_td_tr, reg_gae_tr)

    init_carry = (
        jnp.zeros_like(val_deltas_T[0]),
        jnp.zeros_like(val_deltas_T[0]),
        jnp.zeros_like(reg_deltas_T[0]),
        jnp.zeros_like(reg_deltas_T[0]),
    )
    _, (td_T, gae_T, reg_td_T, reg_gae_T) = jax.lax.scan(
        scan_fn, init_carry, (val_deltas_T, reg_deltas_T), reverse=True
    )

    val_est_T = td_T + vp_T
    scalar_gae_T = jnp.matmul(gae_T, CATEGORICAL_VALUES) + reg_temp * reg_gae_T
    reg_ret_T = (reg_td_T + ents_T) / reg_norm

    returns = jnp.swapaxes(val_est_T, 0, 1)       # (N, 40, 3)
    reg_returns = jnp.swapaxes(reg_ret_T, 0, 1)   # (N, 40)
    advantages = jnp.swapaxes(scalar_gae_T, 0, 1) # (N, 40)
    return returns, reg_returns, advantages


def move_loss_fn(
    model: MoveTransformer,
    traj: Transition,
    returns: jax.Array,
    advantages: jax.Array,
    temperature: jax.Array,
    cfg: TrainConfig,
):
    """PPO + categorical value + behavior KL + magnet KL loss with top-quantile advantage filtering."""
    T, B = traj.action.shape
    flat_obs = traj.infostate.reshape(T * B, *traj.infostate.shape[2:])
    flat_mask = traj.legal_mask.reshape(T * B, -1)
    flat_act = traj.action.reshape(T * B)
    flat_old_logp = traj.log_probs.reshape(T * B, -1)
    flat_ret = returns.reshape(T * B, N_VF_CAT)
    flat_adv = advantages.reshape(T * B)

    # Top-quantile advantage filtering (matches `CircularBuffer.process_data` in `pyengine/core/buffer.py`)
    abs_adv = jnp.abs(flat_adv)
    thresh = jnp.maximum(jnp.quantile(abs_adv, cfg.adv_filt_rate), cfg.adv_filt_thresh)
    adv_mask = (abs_adv >= thresh).astype(jnp.float32)
    # Fallback to allones if no sample exceeds threshold
    adv_mask = jnp.where(jnp.sum(adv_mask) > 0, adv_mask, jnp.ones_like(adv_mask))
    denom = jnp.maximum(jnp.sum(adv_mask), 1.0)

    log_probs, val_log_probs = model(flat_obs, flat_mask)

    act_logp = jnp.take_along_axis(log_probs, flat_act[:, None], axis=-1).squeeze(-1)
    old_act_logp = jnp.take_along_axis(flat_old_logp, flat_act[:, None], axis=-1).squeeze(-1)
    ratio = jnp.exp(act_logp - old_act_logp)

    surr1 = flat_adv * ratio
    surr2 = flat_adv * jnp.clip(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
    policy_loss = -jnp.sum(jnp.minimum(surr1, surr2) * adv_mask) / denom

    probs = jnp.exp(log_probs) * flat_mask
    kl_loss = jnp.sum(jnp.sum(probs * (log_probs - flat_old_logp), axis=-1) * adv_mask) / denom

    value_loss = -jnp.sum(jnp.sum(flat_ret * val_log_probs, axis=-1) * adv_mask) / denom

    # Magnet KL to uniform or piece-weighted uniform legal policy
    if cfg.uniform_magnet:
        n_legal = jnp.maximum(jnp.sum(flat_mask.astype(jnp.float32), axis=-1, keepdims=True), 1.0)
        log_magnet = jnp.where(flat_mask, -jnp.log(n_legal), 0.0)
    else:
        magnet = get_weighted_uniform_policy(flat_mask)
        log_magnet = jnp.where(flat_mask, jnp.log(jnp.maximum(magnet, 1e-10)), 0.0)
    magnet_kl = jnp.sum(jnp.sum(probs * (log_probs - log_magnet), axis=-1) * adv_mask) / denom

    total_loss = (
        cfg.policy_coef * policy_loss
        + cfg.vf_coef * value_loss
        + cfg.kl_coef * kl_loss
        + temperature * magnet_kl
    )
    metrics = {
        "move/loss": total_loss,
        "move/policy_loss": policy_loss,
        "move/value_loss": value_loss,
        "move/kl_loss": kl_loss,
        "move/magnet_kl": magnet_kl,
        "move/effective_frac": jnp.mean(adv_mask),
    }
    return total_loss, metrics


def arrangement_loss_fn(
    arr_model: ArrangementTransformer,
    arr_batch: ArrEpisodeBatch,
    ep_return_pl0: jax.Array,  # (B,)
    ep_valid: jax.Array,       # (B,) bool: whether this env completed at least 1 game
    arr_temperature: jax.Array,
    cfg: TrainConfig,
):
    """PPO + categorical value + future entropy prediction loss on initial arrangements."""
    B = ep_return_pl0.shape[0]
    pl_rewards = jnp.stack([ep_return_pl0, -ep_return_pl0], axis=1)  # (B, 2)
    ret_cat = jax.nn.one_hot(
        jnp.round(pl_rewards).astype(jnp.int32) + 1, N_VF_CAT, dtype=jnp.float32
    ).reshape(B * 2, N_VF_CAT)

    flat_oh = arr_batch.canonical_onehot.reshape(B * 2, ARRANGEMENT_SIZE, N_PIECE_TYPE)
    flat_old_logp = arr_batch.old_log_probs.reshape(B * 2, ARRANGEMENT_SIZE, N_PIECE_TYPE)
    flat_old_val = arr_batch.old_values.reshape(B * 2, ARRANGEMENT_SIZE, N_VF_CAT)
    flat_old_ent = arr_batch.old_ent_preds.reshape(B * 2, ARRANGEMENT_SIZE)
    flat_mask = jnp.repeat(ep_valid[:, None], 2, axis=1).reshape(B * 2, 1)  # (B*2, 1)
    denom = jnp.maximum(jnp.sum(flat_mask.astype(jnp.float32)) * ARRANGEMENT_SIZE, 1.0)

    returns, reg_returns, adv = compute_arrangement_prefix_gae(
        flat_oh,
        flat_old_logp,
        flat_old_val,
        flat_old_ent,
        ret_cat,
        cfg.arr_td_lambda,
        cfg.arr_gae_lambda,
        arr_temperature,
        cfg.arr_reg_norm,
    )

    log_probs, val_log_probs, ent_preds = arr_model(flat_oh)
    chosen_idx = jnp.argmax(flat_oh, axis=-1, keepdims=True)
    logp = jnp.take_along_axis(log_probs, chosen_idx, axis=-1).squeeze(-1)
    old_logp = jnp.take_along_axis(flat_old_logp, chosen_idx, axis=-1).squeeze(-1)
    ratio = jnp.exp(logp - old_logp)

    surr1 = adv * ratio
    surr2 = adv * jnp.clip(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
    policy_loss = -jnp.sum(jnp.minimum(surr1, surr2) * flat_mask) / denom
    value_loss = -jnp.sum(jnp.sum(returns * val_log_probs, axis=-1) * flat_mask) / denom
    entropy_loss = jnp.sum(jnp.square(ent_preds - reg_returns) * flat_mask) / denom
    kl_loss = jnp.sum(jnp.sum(jnp.exp(log_probs) * (log_probs - flat_old_logp), axis=-1) * flat_mask) / denom

    total_loss = (
        policy_loss
        + cfg.arr_vf_coef * value_loss
        + cfg.arr_ent_pred_coef * entropy_loss
        + cfg.kl_coef * kl_loss
    )
    return total_loss, {
        "arr/loss": total_loss,
        "arr/value_loss": value_loss,
        "arr/entropy_loss": entropy_loss,
    }


def belief_loss_fn(belief_model: BeliefTransformer, traj: Transition):
    """Cross-entropy loss over unknown opponent pieces (matching `Belief.compute_and_record_loss`)."""
    T, B = traj.action.shape
    N = T * B
    obs = traj.infostate.reshape(N, *traj.infostate.shape[2:])
    pos_oh = traj.unknown_pos_onehot.reshape(N, 40, 100)
    type_oh = traj.unknown_type_onehot.reshape(N, 40, N_PIECE_TYPE)
    moved = traj.unknown_has_moved.reshape(N, 40)
    counts = traj.unknown_counts.reshape(N, 12)

    log_probs = belief_model(obs, pos_oh, type_oh, counts, moved)  # (N, 40, 14)
    active = jnp.any(type_oh, axis=-1)  # (N, 40)
    piece_ce = -jnp.sum(type_oh.astype(jnp.float32) * log_probs, axis=-1)  # (N, 40)
    loss = jnp.sum(piece_ce * active) / jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
    return loss, {"belief/ce_loss": loss}


class TrainState(NamedTuple):
    move_model: MoveTransformer
    ema_move_model: MoveTransformer
    arr_model: ArrangementTransformer
    ema_arr_model: ArrangementTransformer
    belief_model: BeliefTransformer
    move_opt: nnx.Optimizer
    arr_opt: nnx.Optimizer
    belief_opt: nnx.Optimizer
    env_state: EnvState
    arr_batch: ArrEpisodeBatch
    step: jax.Array
    key: jax.Array


def _update_ema(ema_model: nnx.Module, online_model: nnx.Module, decay: float):
    """In-place exponential moving average parameter update."""
    graphdef, ema_state = nnx.split(ema_model)
    _, online_state = nnx.split(online_model)
    new_ema_state = jax.tree.map(
        lambda e, p: decay * e + (1.0 - decay) * p, ema_state, online_state
    )
    nnx.update(ema_model, new_ema_state)


def create_train_state(cfg: TrainConfig, key: jax.Array) -> TrainState:
    k_move, k_arr, k_bel, k_sample, k_env, next_key = jax.random.split(key, 6)
    t_cfg = TransformerConfig(
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        depth=cfg.depth,
        use_bfloat16=cfg.use_bfloat16,
    )
    counts = BARRAGE_COUNTS if cfg.barrage else CLASSIC_COUNTS

    move_model = MoveTransformer(t_cfg, rngs=nnx.Rngs(k_move))
    ema_move_model = MoveTransformer(t_cfg, rngs=nnx.Rngs(k_move))
    arr_model = ArrangementTransformer(counts, t_cfg, rngs=nnx.Rngs(k_arr))
    ema_arr_model = ArrangementTransformer(counts, t_cfg, rngs=nnx.Rngs(k_arr))
    belief_model = BeliefTransformer(t_cfg, rngs=nnx.Rngs(k_bel))

    def _lr_sched(s):
        return power_schedule(cfg.lr_coef, s, cfg.lr_decay, cfg.lr, cfg.lr_floor)

    move_lr_sched = _lr_sched if cfg.use_power_schedule else cfg.lr

    move_tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(move_lr_sched))
    arr_tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.arr_lr if cfg.use_power_schedule else cfg.lr))
    bel_tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))

    move_opt = nnx.Optimizer(move_model, move_tx, wrt=nnx.Param)
    arr_opt = nnx.Optimizer(arr_model, arr_tx, wrt=nnx.Param)
    belief_opt = nnx.Optimizer(belief_model, bel_tx, wrt=nnx.Param)

    # Sample initial arrangements for all environments (2 per env)
    B = cfg.num_envs
    arr_ids, canon_oh, log_p, vals, ent_p = sample_arrangements(arr_model, k_sample, B * 2)
    arr_ids_2 = arr_ids.reshape(B, 2, 40)
    arr_batch = ArrEpisodeBatch(
        canonical_onehot=canon_oh.reshape(B, 2, 40, N_PIECE_TYPE),
        old_log_probs=log_p.reshape(B, 2, 40, N_PIECE_TYPE),
        old_values=vals.reshape(B, 2, 40, N_VF_CAT),
        old_ent_preds=ent_p.reshape(B, 2, 40),
    )

    env_keys = jax.random.split(k_env, B)
    env_state = jax.vmap(lambda k, a0, a1: init_env(k, cfg.barrage, a0, a1))(
        env_keys, arr_ids_2[:, 0], arr_ids_2[:, 1]
    )
    return TrainState(
        move_model=move_model,
        ema_move_model=ema_move_model,
        arr_model=arr_model,
        ema_arr_model=ema_arr_model,
        belief_model=belief_model,
        move_opt=move_opt,
        arr_opt=arr_opt,
        belief_opt=belief_opt,
        env_state=env_state,
        arr_batch=arr_batch,
        step=jnp.int32(0),
        key=next_key,
    )


def make_train_step(cfg: TrainConfig):
    """Build a single JIT-compiled end-to-end rollout + RL + Arrangement + Belief update step."""
    B = cfg.num_envs
    vmap_obs = jax.vmap(lambda s: observe(s, cfg.max_num_moves, cfg.max_moves_between_attacks))
    vmap_mask = jax.vmap(legal_action_mask)
    vmap_step = jax.vmap(lambda s, a: step_env(s, a, cfg.max_num_moves, cfg.max_moves_between_attacks))
    vmap_unknown = jax.vmap(get_unknown_piece_info)
    vmap_init = jax.vmap(lambda k, a0, a1: init_env(k, cfg.barrage, a0, a1))

    @nnx.jit
    def train_step(ts: TrainState) -> tuple[TrainState, dict[str, jax.Array]]:
        k_roll, k_reset_arr, k_reset_env, next_key = jax.random.split(ts.key, 4)

        # Pre-sample fresh arrangements for auto-resets during this rollout window
        fresh_ids, fresh_oh, fresh_logp, fresh_vals, fresh_ents = sample_arrangements(
            ts.arr_model, k_reset_arr, B * 2
        )
        fresh_ids_2 = fresh_ids.reshape(B, 2, 40)
        reset_keys = jax.random.split(k_reset_env, B)
        fresh_env_states = vmap_init(reset_keys, fresh_ids_2[:, 0], fresh_ids_2[:, 1])

        move_graphdef, move_state = nnx.split(ts.move_model)

        def rollout_body(carry, step_key):
            env_s, ep_ret_pl0, ep_finished = carry
            model = nnx.merge(move_graphdef, move_state)

            obs = vmap_obs(env_s)
            mask = vmap_mask(env_s)
            pos_oh, type_oh, moved, counts = vmap_unknown(env_s)

            log_probs, val_log_probs = model(obs, mask)
            actions = jax.random.categorical(step_key, log_probs, axis=-1)

            next_s = vmap_step(env_s, actions)
            newly_done = next_s.done & (~env_s.done)

            acting_sign = jnp.where(env_s.to_play == 0, 1.0, -1.0)
            acting_rew = newly_done.astype(jnp.float32) * next_s.reward_pl0 * acting_sign

            # Record completed game return for arrangement training
            ep_ret_pl0 = jnp.where(newly_done & (~ep_finished), next_s.reward_pl0, ep_ret_pl0)
            ep_finished = ep_finished | newly_done

            # Auto-reset terminated environments to fresh_env_states
            reset_s = jax.tree.map(
                lambda fr, nx: jnp.where(
                    jnp.expand_dims(next_s.done, tuple(range(1, nx.ndim))), fr, nx
                ),
                fresh_env_states,
                next_s,
            )

            trans = Transition(
                infostate=obs,
                legal_mask=mask,
                action=actions,
                log_probs=log_probs,
                value_log_probs=val_log_probs,
                acting_reward=acting_rew,
                newly_done=newly_done,
                unknown_pos_onehot=pos_oh,
                unknown_type_onehot=type_oh,
                unknown_has_moved=moved,
                unknown_counts=counts,
            )
            return (reset_s, ep_ret_pl0, ep_finished), trans

        step_keys = jax.random.split(k_roll, cfg.rollout_steps)
        init_carry = (
            ts.env_state,
            jnp.zeros(B, dtype=jnp.float32),
            jnp.zeros(B, dtype=jnp.bool_),
        )
        (final_env_s, ep_ret_pl0, ep_finished), traj = jax.lax.scan(
            rollout_body, init_carry, step_keys
        )

        # Bootstrap values for player at T and player at T+1
        _, boot_v0 = ts.move_model(vmap_obs(final_env_s), vmap_mask(final_env_s))
        dummy_act = jnp.argmax(vmap_mask(final_env_s).astype(jnp.int32), axis=-1)
        next_final_s = vmap_step(final_env_s, dummy_act)
        _, boot_v1 = ts.move_model(vmap_obs(next_final_s), vmap_mask(next_final_s))

        returns, advantages = compute_two_player_gae(
            traj.value_log_probs,
            traj.acting_reward,
            traj.newly_done,
            boot_v0,
            boot_v1,
            cfg.td_lambda,
            cfg.gae_lambda,
        )

        if cfg.use_power_schedule:
            temperature = power_schedule(
                cfg.temperature_coef,
                ts.step,
                cfg.temperature_decay,
                cfg.temperature_ceil,
                cfg.temperature_floor,
            )
            arr_temperature = power_schedule(
                cfg.arr_temperature_coef,
                ts.step,
                cfg.arr_temperature_decay,
                cfg.arr_temperature_ceil,
                cfg.arr_temperature_floor,
            )
        else:
            temperature = jnp.float32(cfg.temperature)
            arr_temperature = jnp.float32(cfg.arr_temperature)

        # 1. Update MoveTransformer + EMA
        (_, move_metrics), move_grads = nnx.value_and_grad(move_loss_fn, has_aux=True)(
            ts.move_model, traj, returns, advantages, temperature, cfg
        )
        ts.move_opt.update(ts.move_model, move_grads)
        _update_ema(ts.ema_move_model, ts.move_model, cfg.ema_decay)

        # 2. Update ArrangementTransformer + EMA
        (_, arr_metrics), arr_grads = nnx.value_and_grad(arrangement_loss_fn, has_aux=True)(
            ts.arr_model, ts.arr_batch, ep_ret_pl0, ep_finished, arr_temperature, cfg
        )
        ts.arr_opt.update(ts.arr_model, arr_grads)
        _update_ema(ts.ema_arr_model, ts.arr_model, cfg.ema_decay)

        # 3. Update BeliefTransformer
        (_, belief_metrics), belief_grads = nnx.value_and_grad(belief_loss_fn, has_aux=True)(
            ts.belief_model, traj
        )
        ts.belief_opt.update(ts.belief_model, belief_grads)

        # Update stored arrangement batch for envs that reset
        new_arr_batch = ArrEpisodeBatch(
            canonical_onehot=jnp.where(
                ep_finished[:, None, None, None],
                fresh_oh.reshape(B, 2, 40, N_PIECE_TYPE),
                ts.arr_batch.canonical_onehot,
            ),
            old_log_probs=jnp.where(
                ep_finished[:, None, None, None],
                fresh_logp.reshape(B, 2, 40, N_PIECE_TYPE),
                ts.arr_batch.old_log_probs,
            ),
            old_values=jnp.where(
                ep_finished[:, None, None, None],
                fresh_vals.reshape(B, 2, 40, N_VF_CAT),
                ts.arr_batch.old_values,
            ),
            old_ent_preds=jnp.where(
                ep_finished[:, None, None],
                fresh_ents.reshape(B, 2, 40),
                ts.arr_batch.old_ent_preds,
            ),
        )

        metrics = {
            **move_metrics,
            **arr_metrics,
            **belief_metrics,
            "env/finished_games": jnp.sum(traj.newly_done.astype(jnp.int32)),
        }
        new_ts = TrainState(
            move_model=ts.move_model,
            ema_move_model=ts.ema_move_model,
            arr_model=ts.arr_model,
            ema_arr_model=ts.ema_arr_model,
            belief_model=ts.belief_model,
            move_opt=ts.move_opt,
            arr_opt=ts.arr_opt,
            belief_opt=ts.belief_opt,
            env_state=final_env_s,
            arr_batch=new_arr_batch,
            step=ts.step + 1,
            key=next_key,
        )
        return new_ts, metrics

    return train_step
