"""CLI entry point for end-to-end JAX + Flax NNX Stratego training, evaluation, and search."""

import argparse
import time
import jax
import jax.numpy as jnp
from flax import nnx

from jax_stratego.env import init_env, legal_action_mask, observe, step_env
from jax_stratego.networks import ArrangementTransformer, MoveTransformer, sample_arrangements
from jax_stratego.search import SearchConfig, search_action
from jax_stratego.train import TrainConfig, create_train_state, make_train_step


@nnx.jit(static_argnames=("barrage", "num_games", "max_moves", "max_moves_between_attacks"))
def evaluate_vs_random(
    move_model: MoveTransformer,
    arr_model: ArrangementTransformer,
    key: jax.Array,
    barrage: bool = True,
    num_games: int = 128,
    max_moves: int = 400,
    max_moves_between_attacks: int = 80,
) -> dict[str, jax.Array]:
    """Evaluate trained MoveTransformer + ArrangementTransformer against a uniform-random opponent.

    Plays `num_games // 2` games as Player 0 and `num_games // 2` games as Player 1.
    """
    half = num_games // 2
    k_arr0, k_arr1, k_env0, k_env1, k_play0, k_play1 = jax.random.split(key, 6)

    vmap_obs = jax.vmap(lambda s: observe(s, max_moves, max_moves_between_attacks))
    vmap_mask = jax.vmap(legal_action_mask)
    vmap_step = jax.vmap(lambda s, a: step_env(s, a, max_moves, max_moves_between_attacks))
    move_graphdef, move_state = nnx.split(move_model)

    def play_match(env_states, agent_player: int, play_key: jax.Array):
        step_keys = jax.random.split(play_key, max_moves)

        def step_fn(s, k):
            model = nnx.merge(move_graphdef, move_state)
            obs = vmap_obs(s)
            mask = vmap_mask(s)
            model_logp, _ = model(obs, mask)
            rand_logp = jnp.where(mask, 0.0, -1e9)
            is_agent = s.to_play == agent_player
            logits = jnp.where(is_agent[:, None], model_logp, rand_logp)
            actions = jax.random.categorical(k, logits, axis=-1)
            next_s = vmap_step(s, actions)
            return next_s, None

        final_s, _ = jax.lax.scan(step_fn, env_states, step_keys)
        sign = 1.0 if agent_player == 0 else -1.0
        return final_s.reward_pl0 * sign

    # Half 1: Agent is Player 0, Random is Player 1
    agent_arr0, _, _, _, _ = sample_arrangements(arr_model, k_arr0, half)
    env_s0 = jax.vmap(lambda k, a0: init_env(k, barrage=barrage, arr_pl0=a0, arr_pl1=None))(
        jax.random.split(k_env0, half), agent_arr0
    )
    ret0 = play_match(env_s0, 0, k_play0)

    # Half 2: Random is Player 0, Agent is Player 1
    agent_arr1, _, _, _, _ = sample_arrangements(arr_model, k_arr1, half)
    env_s1 = jax.vmap(lambda k, a1: init_env(k, barrage=barrage, arr_pl0=None, arr_pl1=a1))(
        jax.random.split(k_env1, half), agent_arr1
    )
    ret1 = play_match(env_s1, 1, k_play1)

    all_ret = jnp.concatenate([ret0, ret1], axis=0)
    wins = jnp.mean((all_ret > 0.5).astype(jnp.float32))
    ties = jnp.mean((jnp.abs(all_ret) <= 0.5).astype(jnp.float32))
    losses = jnp.mean((all_ret < -0.5).astype(jnp.float32))
    return {
        "eval/return": jnp.mean(all_ret),
        "eval/win_rate": wins,
        "eval/tie_rate": ties,
        "eval/loss_rate": losses,
    }


@nnx.jit(static_argnames=("barrage", "num_games", "max_moves", "max_moves_between_attacks", "num_rollout_envs", "search_depth"))
def evaluate_search_vs_policy(
    move_model: MoveTransformer,
    belief_model,
    arr_model: ArrangementTransformer,
    key: jax.Array,
    barrage: bool = True,
    num_games: int = 16,
    max_moves: int = 200,
    max_moves_between_attacks: int = 60,
    num_rollout_envs: int = 32,
    search_depth: int = 2,
) -> dict[str, jax.Array]:
    """Evaluate SearchBot (with belief determinization) vs raw MoveTransformer policy head-to-head."""
    half = num_games // 2
    k_arr0, k_arr1, k_env0, k_env1, k_play0, k_play1 = jax.random.split(key, 6)
    search_cfg = SearchConfig(
        num_rollout_envs=num_rollout_envs,
        depth=search_depth,
        max_num_Moves=max_moves,
        max_moves_between_attacks=max_moves_between_attacks,
        max_unknown=8 if barrage else 40,
    )

    vmap_obs = jax.vmap(lambda s: observe(s, max_moves, max_moves_between_attacks))
    vmap_mask = jax.vmap(legal_action_mask)
    vmap_step = jax.vmap(lambda s, a: step_env(s, a, max_moves, max_moves_between_attacks))
    move_graphdef, move_state = nnx.split(move_model)
    bel_graphdef, bel_state = nnx.split(belief_model)

    def play_match(env_states, search_player: int, play_key: jax.Array):
        step_keys = jax.random.split(play_key, max_moves)

        def step_fn(s, k):
            m_model = nnx.merge(move_graphdef, move_state)
            b_model = nnx.merge(bel_graphdef, bel_state)
            k_search, k_pol = jax.random.split(k)

            # Policy actions
            obs = vmap_obs(s)
            mask = vmap_mask(s)
            pol_logp, _ = m_model(obs, mask)
            pol_actions = jax.random.categorical(k_pol, pol_logp, axis=-1)

            # SearchBot actions (vectorized across the batch of games)
            s_keys = jax.random.split(k_search, half)
            search_actions, _, _ = jax.vmap(
                lambda st, sk: search_action(m_model, b_model, st, sk, search_cfg)
            )(s, s_keys)

            is_search = s.to_play == search_player
            actions = jnp.where(is_search, search_actions, pol_actions)
            next_s = vmap_step(s, actions)
            return next_s, None

        final_s, _ = jax.lax.scan(step_fn, env_states, step_keys)
        sign = 1.0 if search_player == 0 else -1.0
        return final_s.reward_pl0 * sign

    arr0, _, _, _, _ = sample_arrangements(arr_model, k_arr0, half * 2)
    arr0_2 = arr0.reshape(half, 2, 40)
    env_s0 = jax.vmap(lambda k, a0, a1: init_env(k, barrage=barrage, arr_pl0=a0, arr_pl1=a1))(
        jax.random.split(k_env0, half), arr0_2[:, 0], arr0_2[:, 1]
    )
    ret0 = play_match(env_s0, 0, k_play0)

    arr1, _, _, _, _ = sample_arrangements(arr_model, k_arr1, half * 2)
    arr1_2 = arr1.reshape(half, 2, 40)
    env_s1 = jax.vmap(lambda k, a0, a1: init_env(k, barrage=barrage, arr_pl0=a0, arr_pl1=a1))(
        jax.random.split(k_env1, half), arr1_2[:, 0], arr1_2[:, 1]
    )
    ret1 = play_match(env_s1, 1, k_play1)

    all_ret = jnp.concatenate([ret0, ret1], axis=0)
    wins = jnp.mean((all_ret > 0.5).astype(jnp.float32))
    ties = jnp.mean((jnp.abs(all_ret) <= 0.5).astype(jnp.float32))
    losses = jnp.mean((all_ret < -0.5).astype(jnp.float32))
    return {
        "search_vs_pol/return": jnp.mean(all_ret),
        "search_vs_pol/win_rate": wins,
        "search_vs_pol/tie_rate": ties,
        "search_vs_pol/loss_rate": losses,
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal JAX + Flax NNX Stratego (Ataraxos)")
    parser.add_argument("--classic", action="store_true", help="Use Classic 40-piece Stratego (default: Barrage)")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--rollout_steps", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bfloat16", action="store_true", help="Use bfloat16 Transformer compute")
    parser.add_argument("--power_schedule", action="store_true", help="Use paper power-law LR and temperature schedules")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    max_moves = 400 if not args.classic else 1000
    max_moves_attack = 80 if not args.classic else 150
    cfg = TrainConfig(
        barrage=not args.classic,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        max_num_moves=max_moves,
        max_moves_between_attacks=max_moves_attack,
        d_model=args.d_model,
        depth=args.depth,
        lr=args.lr,
        use_bfloat16=args.bfloat16,
        use_power_schedule=args.power_schedule,
    )
    print(
        f"Devices: {jax.devices()} | Variant: {'Barrage' if cfg.barrage else 'Classic'} | "
        f"Envs: {cfg.num_envs} | Rollout: {cfg.rollout_steps} | Iters: {args.iterations}",
        flush=True,
    )
    key = jax.random.PRNGKey(args.seed)
    k_init, k_eval, k_search = jax.random.split(key, 3)
    ts = create_train_state(cfg, k_init)
    train_step = make_train_step(cfg)

    # Initial baseline evaluation (Iter 0)
    k_eval, subk = jax.random.split(k_eval)
    eval0 = evaluate_vs_random(
        ts.move_model, ts.arr_model, subk, barrage=cfg.barrage,
        max_moves=max_moves, max_moves_between_attacks=max_moves_attack,
    )
    print(
        f"[Eval 000] vs_random: return={float(eval0['eval/return']):+.3f} "
        f"(win={float(eval0['eval/win_rate']):.1%}, tie={float(eval0['eval/tie_rate']):.1%}, "
        f"loss={float(eval0['eval/loss_rate']):.1%})",
        flush=True,
    )

    for it in range(1, args.iterations + 1):
        t0 = time.perf_counter()
        ts, metrics = train_step(ts)
        jax.block_until_ready(metrics["move/loss"])
        dt = time.perf_counter() - t0
        if it == 1 or it % 5 == 0 or it == args.iterations:
            print(
                f"[Iter {it:03d}] {dt:.2f}s | "
                f"move_loss={float(metrics['move/loss']):.4f} "
                f"(pol={float(metrics['move/policy_loss']):.4f}, val={float(metrics['move/value_loss']):.4f}) | "
                f"arr_loss={float(metrics['arr/loss']):.4f} | "
                f"belief_ce={float(metrics['belief/ce_loss']):.4f} | "
                f"finished_games={int(metrics['env/finished_games'])}",
                flush=True,
            )
        if it % args.eval_every == 0 or it == args.iterations:
            k_eval, subk = jax.random.split(k_eval)
            ev = evaluate_vs_random(
                ts.ema_move_model, ts.ema_arr_model, subk, barrage=cfg.barrage,
                max_moves=max_moves, max_moves_between_attacks=max_moves_attack,
            )
            print(
                f"[Eval {it:03d}] vs_random: return={float(ev['eval/return']):+.3f} "
                f"(win={float(ev['eval/win_rate']):.1%}, tie={float(ev['eval/tie_rate']):.1%}, "
                f"loss={float(ev['eval/loss_rate']):.1%})",
                flush=True,
            )

    # Final head-to-head evaluation: SearchBot (with learned BeliefTransformer) vs raw Policy Network
    t0 = time.perf_counter()
    s_ev = evaluate_search_vs_policy(
        ts.ema_move_model, ts.belief_model, ts.ema_arr_model, k_search,
        barrage=cfg.barrage, max_moves=max_moves, max_moves_between_attacks=max_moves_attack,
    )
    jax.block_until_ready(s_ev["search_vs_pol/return"])
    dt = time.perf_counter() - t0
    print(
        f"[SearchBot vs Policy] {dt:.2f}s | return={float(s_ev['search_vs_pol/return']):+.3f} "
        f"(win={float(s_ev['search_vs_pol/win_rate']):.1%}, tie={float(s_ev['search_vs_pol/tie_rate']):.1%}, "
        f"loss={float(s_ev['search_vs_pol/loss_rate']):.1%})",
        flush=True,
    )


if __name__ == "__main__":
    main()
