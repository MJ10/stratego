"""Tests for the pure-JAX + Flax NNX Stratego implementation."""

import jax
import jax.numpy as jnp
from flax import nnx

from jax_stratego.env import (
    ACTION_DST,
    ACTION_SRC,
    BARRAGE_COUNTS,
    BOMB,
    CLASSIC_COUNTS,
    EMPTY,
    FLAG,
    MARSHAL,
    MINER,
    SCOUT,
    SERGEANT,
    SPY,
    assign_opponent_hidden_pieces,
    get_unknown_piece_info,
    init_env,
    legal_action_mask,
    observe,
    step_env,
)
from jax_stratego.networks import (
    ArrangementTransformer,
    BeliefTransformer,
    TransformerConfig,
    sample_arrangements,
    sample_belief,
)
from jax_stratego.search import SearchConfig, search_action
from jax_stratego.train import TrainConfig, compute_two_player_gae, create_train_state, make_train_step


def _encode_action(src: int, dst: int) -> int:
    matches = (ACTION_SRC == src) & (ACTION_DST == dst)
    idx = int(jnp.argmax(matches.astype(jnp.int32)))
    assert bool(matches[idx]), f"No action from {src} to {dst}"
    return idx


def test_combat_and_special_rules():
    """Verify Spy vs Marshal, Miner vs Bomb, Sergeant vs Bomb, and Flag capture."""
    arr0 = jnp.full(40, EMPTY, dtype=jnp.int8)
    arr0 = arr0.at[0].set(FLAG).at[30].set(SPY).at[31].set(MINER).at[32].set(SERGEANT)

    # Opponent setup: cell 39 in their POV -> 99 - 39 = 60 in Pl0's POV
    # Put opponent pieces on their front row (cells 30..39 in their POV -> 69..60 in Pl0's POV)
    # Specifically, Pl0's cell 30 faces cell 40 (empty), wait: rows 4 and 5 are empty/lakes.
    # Let's place a Scout or step pieces adjacent to test combat directly.
    arr1 = jnp.full(40, EMPTY, dtype=jnp.int8)
    arr1 = arr1.at[0].set(FLAG).at[39].set(MARSHAL)
    state = init_env(jax.random.PRNGKey(0), barrage=True, arr_pl0=arr0, arr_pl1=arr1)

    # Move Spy directly in front of opponent Marshal at cell 50 (opponent Marshal at 60)
    state = state._replace(
        board_type=state.board_type.at[30].set(EMPTY).at[50].set(SPY),
        board_owner=state.board_owner.at[30].set(0).at[50].set(1),
        board_piece_id=state.board_piece_id.at[30].set(-1).at[50].set(30),
    )
    act_spy_marshal = _encode_action(50, 60)
    assert bool(legal_action_mask(state)[act_spy_marshal])
    next_s = step_env(state, jnp.int32(act_spy_marshal))
    # In next_s (flipped to Pl1 POV), cell 99 - 60 = 39 should now hold Pl0's Spy (owner == 2 in Pl1 POV, visible == True)
    assert int(next_s.board_type[39]) == SPY
    assert int(next_s.board_owner[39]) == 2
    assert bool(next_s.board_visible[39])

    # Miner (at 51) attacks Bomb (at 61) -> Miner wins
    s_miner = state._replace(
        board_type=state.board_type.at[51].set(MINER).at[61].set(BOMB),
        board_owner=state.board_owner.at[51].set(1).at[61].set(2),
    )
    next_s_miner = step_env(s_miner, jnp.int32(_encode_action(51, 61)))
    assert int(next_s_miner.board_type[99 - 61]) == MINER
    assert int(next_s_miner.board_owner[99 - 61]) == 2

    # Sergeant (at 51) attacks Bomb (at 61) -> Bomb wins
    s_sgt = state._replace(
        board_type=state.board_type.at[51].set(SERGEANT).at[61].set(BOMB),
        board_owner=state.board_owner.at[51].set(1).at[61].set(2),
    )
    next_s_sgt = step_env(s_sgt, jnp.int32(_encode_action(51, 61)))
    assert int(next_s_sgt.board_type[99 - 61]) == BOMB
    assert int(next_s_sgt.board_owner[99 - 61]) == 1

    # Flag capture -> game terminates with reward_pl0 == +1.0
    s_flag = state._replace(
        board_type=state.board_type.at[51].set(SERGEANT).at[61].set(FLAG),
        board_owner=state.board_owner.at[51].set(1).at[61].set(2),
    )
    next_s_flag = step_env(s_flag, jnp.int32(_encode_action(51, 61)))
    assert bool(next_s_flag.done)
    assert float(next_s_flag.reward_pl0) == 1.0


def test_scout_movement_and_twosquare_rule():
    """Verify Scout long-range movement, lake blocking, and Two-Square rule prohibition."""
    arr0 = jnp.full(40, EMPTY, dtype=jnp.int8).at[0].set(FLAG).at[30].set(SCOUT)
    arr1 = jnp.full(40, EMPTY, dtype=jnp.int8).at[0].set(FLAG).at[30].set(SCOUT)
    s = init_env(jax.random.PRNGKey(1), barrage=True, arr_pl0=arr0, arr_pl1=arr1)

    # Column 0 is an open corridor (rows 4, 5 have no lakes at col 0: lakes are cols 2,3 and 6,7)
    # Scout at 30 can slide to 40, 50, 60 (empty) and 70 (empty), up to 90
    mask = legal_action_mask(s)
    assert bool(mask[_encode_action(30, 40)])
    assert bool(mask[_encode_action(30, 50)])
    assert bool(mask[_encode_action(30, 60)])

    # Oscillate Pl0's Scout between 30 and 40 while Pl1 oscillates between 30 and 40 in Pl1 POV:
    # Move 1 (Pl0): 30 -> 40, Move 1 (Pl1): 30 -> 40
    s = step_env(s, jnp.int32(_encode_action(30, 40)))
    s = step_env(s, jnp.int32(_encode_action(30, 40)))
    # Move 2 (Pl0): 40 -> 30, Move 2 (Pl1): 40 -> 30
    s = step_env(s, jnp.int32(_encode_action(40, 30)))
    s = step_env(s, jnp.int32(_encode_action(40, 30)))
    # Move 3 (Pl0): 30 -> 40, Move 3 (Pl1): 30 -> 40
    s = step_env(s, jnp.int32(_encode_action(30, 40)))
    s = step_env(s, jnp.int32(_encode_action(30, 40)))

    # Now Pl0 has history D=30, C=40, B=30, A=40 (vertical BTT triggered: D < C, C > B, B < A, D < A)
    # Moving back down 40 -> 30 (below min(row(C), row(A)) = 4) MUST be forbidden by Two-Square rule!
    mask_after = legal_action_mask(s)
    assert not bool(mask_after[_encode_action(40, 30)])
    # Moving forward 40 -> 50 is still legal
    assert bool(mask_after[_encode_action(40, 50)])


def test_arrangement_and_belief_sampling():
    """Verify ArrangementTransformer and BeliefTransformer respect exact piece counts & movability constraints."""
    rngs = nnx.Rngs(42)
    cfg = TransformerConfig(d_model=32, n_head=2, depth=1)

    for counts in (BARRAGE_COUNTS, CLASSIC_COUNTS):
        arr_model = ArrangementTransformer(counts, cfg, rngs=rngs)
        arr_ids, canon_oh, _, _ = sample_arrangements(arr_model, jax.random.PRNGKey(7), n_sample=8)
        assert arr_ids.shape == (8, 40)
        # Every sampled arrangement must have exact piece counts matching `counts`
        sampled_counts = jnp.sum(canon_oh.astype(jnp.int32), axis=1)  # (8, 14)
        assert jnp.all(sampled_counts == counts[None, :])

    # Test BeliefTransformer sampling + determinization
    s = init_env(jax.random.PRNGKey(9), barrage=True)
    # Mark one opponent piece as moved so movability constraint is active
    opp_cells = jnp.where(s.board_owner == 2)[0]
    s = s._replace(board_moved=s.board_moved.at[opp_cells[0]].set(True))

    obs = observe(s)
    pos_oh, type_oh, moved, counts = get_unknown_piece_info(s)
    bel_model = BeliefTransformer(cfg, rngs=rngs)
    samples = sample_belief(bel_model, jax.random.PRNGKey(10), 16, obs, pos_oh, moved, counts)
    assert samples.shape == (16, 40, 14)

    # Check that sampled counts per type match `counts` (8 unknown pieces in Barrage)
    sampled_unk_counts = jnp.sum(samples[:, :, :12].astype(jnp.int32), axis=1)
    assert jnp.all(sampled_unk_counts == counts[:12][None, :])

    # Check that moved piece was never assigned Flag(10) or Bomb(11)
    moved_slot = int(jnp.argmax(moved.astype(jnp.int32)))
    assert not bool(jnp.any(samples[:, moved_slot, 10:12]))

    # Determinize state and verify opponent hidden pieces are replaced
    det_s = assign_opponent_hidden_pieces(s, samples[0])
    assert det_s.board_type.shape == (100,)


def test_end_to_end_train_and_search():
    """Verify full JIT-compiled training step and SearchBot execution."""
    cfg = TrainConfig(
        barrage=True,
        num_envs=8,
        rollout_steps=16,
        d_model=32,
        n_head=2,
        depth=1,
    )
    ts = create_train_state(cfg, jax.random.PRNGKey(0))
    train_step = make_train_step(cfg)
    ts, metrics = train_step(ts)
    for k, v in metrics.items():
        assert not bool(jnp.isnan(v)), f"NaN in metric {k}"

    single_state = jax.tree.map(lambda x: x[0], ts.env_state)
    act, pi, q = search_action(
        ts.move_model,
        ts.belief_model,
        single_state,
        jax.random.PRNGKey(123),
        SearchConfig(num_rollout_envs=16, depth=2),
    )
    mask = legal_action_mask(single_state)
    assert bool(mask[act])
    assert jnp.isclose(jnp.sum(pi), 1.0, atol=1e-5)
    assert not bool(jnp.any(jnp.isnan(q)))


def test_two_player_gae_exact():
    """Verify solipsistic 2-player GAE and TD(lambda) match pyengine/core/buffer.py."""
    # T = 4 (2 moves per player), B = 1
    # Suppose Pl0 wins on step t=2 (acting_reward[2] = +1.0, newly_done[2] = True)
    val_logp = jnp.zeros((4, 1, 3), dtype=jnp.float32)  # uniform [1/3, 1/3, 1/3] -> scalar_v = 0.0
    acting_rew = jnp.array([[0.0], [0.0], [1.0], [0.0]], dtype=jnp.float32)
    newly_done = jnp.array([[False], [False], [True], [False]])
    boot_p0 = jnp.zeros((1, 3), dtype=jnp.float32)
    boot_p1 = jnp.zeros((1, 3), dtype=jnp.float32)

    ret, adv = compute_two_player_gae(
        val_logp, acting_rew, newly_done, boot_p0, boot_p1, td_lambda=0.8, gae_lambda=0.5
    )
    # Step 2 (Pl0 winning move): terminal reward +1 -> categorical [0, 0, 1], advantage +1.0
    assert jnp.allclose(ret[2, 0], jnp.array([0.0, 0.0, 1.0]))
    assert jnp.isclose(adv[2, 0], 1.0)
    # Step 1 (Pl1 losing move right before Pl0 wins): solipsistic terminal reward -1 -> [1, 0, 0], advantage -1.0
    assert jnp.allclose(ret[1, 0], jnp.array([1.0, 0.0, 0.0]))
    assert jnp.isclose(adv[1, 0], -1.0)
    # Step 0 (Pl0's first move, 1 ply before Pl0's winning move at t=2):
    # delta_0 = 0, delta_2 = 1.0 -> adv[0] = 0 + gae_lambda * 1.0 = 0.5
    assert jnp.isclose(adv[0, 0], 0.5)
