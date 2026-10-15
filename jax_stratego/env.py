"""Pure-JAX Stratego environment (Barrage & Classic) with Two-Square rule and belief state."""

from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp

# Board & Piece constants (matching pystratego)
BOARD_LEN = 10
N_BOARD_CELL = 100
LAKE_INDICES = jnp.array([42, 43, 46, 47, 52, 53, 56, 57], dtype=jnp.int32)
OCCUPIABLE_MASK = jnp.ones(N_BOARD_CELL, dtype=jnp.bool_).at[LAKE_INDICES].set(False)
OCCUPIABLE_INDICES = jnp.where(OCCUPIABLE_MASK)[0]  # (92,)
N_OCCUPIABLE_CELL = 92

N_PLAYER = 2
N_PIECE_TYPE = 14
N_PLAYER_PIECE_TYPE = 12
ARRANGEMENT_SIZE = 40
N_ACTION = 1800
MOVE_MEMORY = 16

SPY, SCOUT, MINER, SERGEANT, LIEUTENANT, CAPTAIN, MAJOR, COLONEL, GENERAL, MARSHAL = range(10)
FLAG, BOMB, LAKE, EMPTY = 10, 11, 12, 13

CLASSIC_COUNTS = jnp.array([1, 8, 5, 4, 4, 4, 3, 2, 1, 1, 1, 6, 0, 0], dtype=jnp.int32)
BARRAGE_COUNTS = jnp.array([1, 2, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 32], dtype=jnp.int32)

# Precompute action <-> (src_cell, dst_cell) tables in acting-player POV
def _build_action_tables():
    src_cells = jnp.arange(N_ACTION, dtype=jnp.int32) % 100
    src_r = src_cells // 10
    src_c = src_cells % 10
    coord = (jnp.arange(N_ACTION, dtype=jnp.int32) // 100) % 9
    horiz = jnp.arange(N_ACTION, dtype=jnp.int32) >= 900
    dst_r = jnp.where(horiz, src_r, coord + (coord >= src_r).astype(jnp.int32))
    dst_c = jnp.where(horiz, coord + (coord >= src_c).astype(jnp.int32), src_c)
    dst_cells = dst_r * 10 + dst_c

    cell_to_occ = jnp.full(N_BOARD_CELL, -1, dtype=jnp.int32).at[OCCUPIABLE_INDICES].set(
        jnp.arange(N_OCCUPIABLE_CELL, dtype=jnp.int32)
    )
    src_occ = cell_to_occ[src_cells]
    dst_occ = cell_to_occ[dst_cells]
    valid_occ = (src_occ >= 0) & (dst_occ >= 0)
    flat_occ_idx = jnp.where(valid_occ, src_occ * N_OCCUPIABLE_CELL + dst_occ, 0)
    return src_cells, dst_cells, horiz, valid_occ, flat_occ_idx


ACTION_SRC, ACTION_DST, ACTION_HORIZ, ACTION_VALID_OCC, ACTION_TO_OCC_PAIR = _build_action_tables()


class EnvState(NamedTuple):
    # Board tensors are always stored in the ACTING PLAYER'S POV (rows 0..3 = us at start).
    # owner: 0 = empty, 1 = current player (us), 2 = opponent (them), 3 = lake
    board_type: jax.Array        # int8[100]
    board_owner: jax.Array       # int8[100]
    board_visible: jax.Array     # bool[100]
    board_moved: jax.Array       # bool[100]
    board_piece_id: jax.Array    # int8[100] (0..39 for player pieces, -1 for empty/lake)
    # Initial arrangements for RL arrangement buffer & cemetery features: (2, 40) in [0..13]
    # Index 0 = current acting player, Index 1 = opponent
    init_arrangements: jax.Array  # int8[2, 40]
    # Two-square state per player: (2, 4) storing (A, B, C, D) in each player's own POV (-1 = none)
    twosquare: jax.Array          # int8[2, 4]
    # Recent move history: (MOVE_MEMORY, 100) float32 planes (-1 at src, +1 at dst, in current POV)
    move_history: jax.Array       # float32[MOVE_MEMORY, 100]
    to_play: jax.Array            # int32 scalar (0 or 1)
    num_moves: jax.Array          # int32 scalar
    moves_since_attack: jax.Array # int32 scalar
    done: jax.Array               # bool scalar
    reward_pl0: jax.Array         # float32 scalar


def sample_random_arrangement(key: jax.Array, piece_counts: jax.Array) -> jax.Array:
    """Sample a random valid arrangement of length 40 given piece_counts[14]."""
    # Expand counts into 40 piece IDs statically (using Classic/Barrage max counts)
    piece_ids = jnp.repeat(
        jnp.arange(N_PIECE_TYPE, dtype=jnp.int8),
        piece_counts,
        total_repeat_length=ARRANGEMENT_SIZE,
    )
    perm = jax.random.permutation(key, ARRANGEMENT_SIZE)
    return piece_ids[perm]


def _raw_legal_actions(board_type: jax.Array, board_owner: jax.Array) -> jax.Array:
    """Compute legal action mask (1800,) for owner==1 ignoring two-square rule."""
    bo = board_owner.reshape(10, 10)

    src_r = ACTION_SRC // 10
    src_c = ACTION_SRC % 10
    dst_r = ACTION_DST // 10
    dst_c = ACTION_DST % 10

    p_type = board_type[ACTION_SRC]
    p_owner = board_owner[ACTION_SRC]
    dst_owner = board_owner[ACTION_DST]

    can_piece_move = (p_owner == 1) & (p_type < FLAG)
    valid_target = (dst_owner == 0) | (dst_owner == 2)

    dr = jnp.abs(dst_r - src_r)
    dc = jnp.abs(dst_c - src_c)
    dist = dr + dc

    is_step_1 = dist == 1
    is_scout_slide = (p_type == SCOUT) & (dist > 1)

    # Check intermediate squares for scout slides: all cells strictly between src and dst must be empty (owner == 0)
    steps = jnp.arange(1, 9, dtype=jnp.int32)[:, None]  # (8, 1)
    step_r = src_r[None, :] + jnp.sign(dst_r - src_r)[None, :] * steps
    step_c = src_c[None, :] + jnp.sign(dst_c - src_c)[None, :] * steps
    step_r = jnp.clip(step_r, 0, 9)
    step_c = jnp.clip(step_c, 0, 9)
    inter_owner = bo[step_r, step_c]  # (8, 1800)
    within_path = steps < dist[None, :]
    path_clear = jnp.all(~within_path | (inter_owner == 0), axis=0)

    return can_piece_move & valid_target & (is_step_1 | (is_scout_slide & path_clear))


def _apply_twosquare_mask(mask: jax.Array, ts: jax.Array) -> jax.Array:
    """Remove actions forbidden by the Two-Square rule for the acting player (ts: int8[4])."""
    A, B, C, D = ts[0].astype(jnp.int32), ts[1].astype(jnp.int32), ts[2].astype(jnp.int32), ts[3].astype(jnp.int32)
    valid_d = D >= 0
    col_a, col_b, col_c, col_d = A % 10, B % 10, C % 10, D % 10
    row_a, row_b, row_c, row_d = A // 10, B // 10, C // 10, D // 10

    is_from_a = ACTION_SRC == A
    coord = (jnp.arange(N_ACTION, dtype=jnp.int32) // 100) % 9

    # Horizontal LTR: cannot move left of min(col_c, col_a)
    h_ltr = valid_d & (col_d < col_c) & (col_c > col_b) & (col_b < col_a) & (col_d < col_a)
    forbid_h_ltr = h_ltr & is_from_a & ACTION_HORIZ & (coord < jnp.minimum(col_c, col_a))

    # Horizontal RTL: cannot move right of max(col_c, col_a)
    h_rtl = valid_d & (col_d > col_c) & (col_c < col_b) & (col_b > col_a) & (col_d > col_a)
    forbid_h_rtl = h_rtl & is_from_a & ACTION_HORIZ & (coord >= jnp.maximum(col_c, col_a))

    # Vertical TTB: cannot move below min(row_c, row_a)
    v_ttb = valid_d & (row_d < row_c) & (row_c > row_b) & (row_b < row_a) & (row_d < row_a)
    forbid_v_ttb = v_ttb & is_from_a & (~ACTION_HORIZ) & (coord < jnp.minimum(row_c, row_a))

    # Vertical BTT: cannot move above max(row_c, row_a)
    v_btt = valid_d & (row_d > row_c) & (row_c < row_b) & (row_b > row_a) & (row_d > row_a)
    forbid_v_btt = v_btt & is_from_a & (~ACTION_HORIZ) & (coord >= jnp.maximum(row_c, row_a))

    forbid = forbid_h_ltr | forbid_h_rtl | forbid_v_ttb | forbid_v_btt
    return mask & (~forbid)


def legal_action_mask(state: EnvState) -> jax.Array:
    """Return bool[1800] legal action mask for state."""
    raw = _raw_legal_actions(state.board_type, state.board_owner)
    masked = _apply_twosquare_mask(raw, state.twosquare[0])
    # If terminal or no legal moves, keep action 0 true to avoid NaN softmax downstream
    fallback = jnp.zeros(N_ACTION, dtype=jnp.bool_).at[0].set(True)
    return jnp.where(state.done | (~ jnp.any(masked)), fallback, masked)


def init_env(
    key: jax.Array,
    barrage: bool = True,
    arr_pl0: Optional[jax.Array] = None,
    arr_pl1: Optional[jax.Array] = None,
) -> EnvState:
    """Initialize a Stratego environment state from optional or random arrangements."""
    counts = BARRAGE_COUNTS if barrage else CLASSIC_COUNTS
    k0, k1 = jax.random.split(key)
    if arr_pl0 is None:
        arr_pl0 = sample_random_arrangement(k0, counts)
    if arr_pl1 is None:
        arr_pl1 = sample_random_arrangement(k1, counts)

    # Ensure valid arrangements that have at least one movable piece able to move;
    # construct 100-cell board from Player 0's POV
    board_type = jnp.full(N_BOARD_CELL, EMPTY, dtype=jnp.int8)
    board_owner = jnp.zeros(N_BOARD_CELL, dtype=jnp.int8)
    board_piece_id = jnp.full(N_BOARD_CELL, -1, dtype=jnp.int8)

    # Player 0 (us = 1) occupies cells 0..39
    p0_is_piece = arr_pl0 != EMPTY
    board_type = board_type.at[:40].set(arr_pl0.astype(jnp.int8))
    board_owner = board_owner.at[:40].set(jnp.where(p0_is_piece, 1, 0).astype(jnp.int8))
    board_piece_id = board_piece_id.at[:40].set(
        jnp.where(p0_is_piece, jnp.arange(40, dtype=jnp.int8), -1)
    )

    # Lakes at middle rows
    board_type = board_type.at[LAKE_INDICES].set(LAKE)
    board_owner = board_owner.at[LAKE_INDICES].set(3)

    # Player 1 (them = 2) occupies cells 99..60 (flipped from their 0..39)
    arr_pl1_flipped = jnp.flip(arr_pl1)
    p1_ids_flipped = jnp.flip(jnp.arange(40, dtype=jnp.int8))
    p1_is_piece = arr_pl1_flipped != EMPTY
    board_type = board_type.at[60:100].set(arr_pl1_flipped.astype(jnp.int8))
    board_owner = board_owner.at[60:100].set(jnp.where(p1_is_piece, 2, 0).astype(jnp.int8))
    board_piece_id = board_piece_id.at[60:100].set(
        jnp.where(p1_is_piece, p1_ids_flipped, -1)
    )

    board_visible = (board_owner == 0) | (board_owner == 3)
    board_moved = jnp.zeros(N_BOARD_CELL, dtype=jnp.bool_)
    init_arrangements = jnp.stack([arr_pl0.astype(jnp.int8), arr_pl1.astype(jnp.int8)], axis=0)
    twosquare = jnp.full((2, 4), -1, dtype=jnp.int8)
    move_history = jnp.zeros((MOVE_MEMORY, N_BOARD_CELL), dtype=jnp.float32)

    # Check if either player has no legal move at initialization
    p0_has_move = jnp.any(_raw_legal_actions(board_type, board_owner))
    p1_bo = jnp.flip(jnp.where(board_owner == 1, 2, jnp.where(board_owner == 2, 1, board_owner)))
    p1_bt = jnp.flip(board_type)
    p1_has_move = jnp.any(_raw_legal_actions(p1_bt, p1_bo))

    done = (~p0_has_move) | (~p1_has_move)
    reward_pl0 = jnp.where(
        ~done,
        0.0,
        jnp.where(p0_has_move & (~p1_has_move), 1.0, jnp.where((~p0_has_move) & p1_has_move, -1.0, 0.0)),
    )

    return EnvState(
        board_type=board_type,
        board_owner=board_owner,
        board_visible=board_visible,
        board_moved=board_moved,
        board_piece_id=board_piece_id,
        init_arrangements=init_arrangements,
        twosquare=twosquare,
        move_history=move_history,
        to_play=jnp.int32(0),
        num_moves=jnp.int32(0),
        moves_since_attack=jnp.int32(0),
        done=done,
        reward_pl0=reward_pl0,
    )


def step_env(
    state: EnvState,
    action: jax.Array,
    max_num_moves: int = 2000,
    max_moves_between_attacks: int = 100,
) -> EnvState:
    """Apply action in acting player's POV and flip board to next player's POV."""
    src = ACTION_SRC[action]
    dst = ACTION_DST[action]

    from_type = state.board_type[src]
    from_vis = state.board_visible[src]
    from_id = state.board_piece_id[src]

    to_type = state.board_type[dst]
    to_owner = state.board_owner[dst]
    to_vis = state.board_visible[dst]
    to_id = state.board_piece_id[dst]

    is_attack = to_owner == 2
    step_dist = jnp.abs(src // 10 - dst // 10) + jnp.abs(src % 10 - dst % 10)
    is_scout_leap = step_dist >= 2

    # Visibility updates
    new_from_vis = from_vis | is_attack | is_scout_leap
    new_to_vis = to_vis | is_attack

    # Combat resolution (matches ApplyActionsKernel in action_kernels.cu)
    to_wins = is_attack & (
        ((to_type < FLAG) & (to_type > from_type) & ~((to_type == MARSHAL) & (from_type == SPY)))
        | ((to_type == BOMB) & (from_type != MINER))
    )
    tie = is_attack & (to_type == from_type)
    from_wins = ~to_wins & ~tie
    flag_captured = is_attack & (to_type == FLAG)

    # Resulting piece on dst
    dst_type = jnp.where(to_wins, to_type, jnp.where(tie, jnp.int8(EMPTY), from_type))
    dst_owner = jnp.where(to_wins, jnp.int8(2), jnp.where(tie, jnp.int8(0), jnp.int8(1)))
    dst_vis = jnp.where(to_wins, new_to_vis, jnp.where(tie, True, new_from_vis))
    dst_moved = jnp.where(to_wins, state.board_moved[dst], jnp.where(tie, False, True))
    dst_id = jnp.where(to_wins, to_id, jnp.where(tie, jnp.int8(-1), from_id))

    board_type = state.board_type.at[src].set(EMPTY).at[dst].set(dst_type)
    board_owner = state.board_owner.at[src].set(0).at[dst].set(dst_owner)
    board_visible = state.board_visible.at[src].set(True).at[dst].set(dst_vis)
    board_moved = state.board_moved.at[src].set(False).at[dst].set(dst_moved)
    board_piece_id = state.board_piece_id.at[src].set(-1).at[dst].set(dst_id)

    # Update Two-Square rule state for acting player (index 0) and opponent (index 1)
    ts0 = state.twosquare[0]
    A, B, C, _ = ts0[0], ts0[1], ts0[2], ts0[3]
    is_vert = ~ACTION_HORIZ[action]
    last_vert = (A % 10) == (B % 10)
    chain_cont = (A == src) & (is_vert == last_vert)
    new_D = jnp.where(chain_cont, C, jnp.int8(-1))
    new_C = jnp.where(chain_cont, B, jnp.int8(-1))
    new_A = dst.astype(jnp.int8)
    new_B = src.astype(jnp.int8)
    ts0_updated = jnp.array([new_A, new_B, new_C, new_D], dtype=jnp.int8)
    # If our moving piece died on dst (to_wins or tie), clear our twosquare state
    ts0_updated = jnp.where(to_wins | tie, jnp.full(4, -1, dtype=jnp.int8), ts0_updated)

    # If opponent's piece on dst died (from_wins or tie during attack), check if opponent's A == 99 - dst
    ts1 = state.twosquare[1]
    opp_died_at_a = is_attack & (from_wins | tie) & (ts1[0] == (99 - dst))
    ts1_updated = jnp.where(opp_died_at_a, jnp.full(4, -1, dtype=jnp.int8), ts1)

    # Record move in history (in current player's POV before flipping)
    move_plane = jnp.zeros(N_BOARD_CELL, dtype=jnp.float32).at[src].set(-1.0).at[dst].set(1.0)
    # Flip all history planes by 180 degrees so they are in the NEXT player's POV
    next_move_history = jnp.roll(state.move_history, shift=1, axis=0).at[0].set(move_plane)
    next_move_history = jnp.flip(next_move_history, axis=-1)

    # Flip board to next player's POV
    next_board_type = jnp.flip(board_type)
    next_board_owner = jnp.flip(
        jnp.where(board_owner == 1, jnp.int8(2), jnp.where(board_owner == 2, jnp.int8(1), board_owner))
    )
    next_board_visible = jnp.flip(board_visible)
    next_board_moved = jnp.flip(board_moved)
    next_board_piece_id = jnp.flip(board_piece_id)
    next_init_arrs = jnp.flip(state.init_arrangements, axis=0)
    next_twosquare = jnp.stack([ts1_updated, ts0_updated], axis=0)

    next_num_moves = state.num_moves + 1
    next_moves_since_attack = jnp.where(is_attack, jnp.int32(0), state.moves_since_attack + 1)
    next_to_play = 1 - state.to_play

    # Check legal movements for both players (Matches ComputeRewardPl0Kernel in termination_kernels.cu)
    next_pl_mask = _apply_twosquare_mask(_raw_legal_actions(next_board_type, next_board_owner), next_twosquare[0])
    next_pl_can_move = jnp.any(next_pl_mask)

    curr_pl_mask = _apply_twosquare_mask(_raw_legal_actions(board_type, board_owner), ts0_updated)
    curr_pl_can_move = jnp.any(curr_pl_mask)

    is_timeout = (next_num_moves >= max_num_moves) | (next_moves_since_attack >= max_moves_between_attacks)
    no_movement = (~next_pl_can_move) | (~curr_pl_can_move)
    newly_done = flag_captured | is_timeout | no_movement

    # Compute reward from Acting Player's perspective first, then convert to Pl0 sign
    acting_sign = jnp.where(state.to_play == 0, 1.0, -1.0)
    no_move_acting_reward = jnp.where(
        curr_pl_can_move & (~next_pl_can_move),
        1.0,
        jnp.where((~curr_pl_can_move) & next_pl_can_move, -1.0, 0.0),
    )
    acting_reward = jnp.where(
        is_timeout,
        0.0,
        jnp.where(flag_captured, 1.0, jnp.where(no_movement, no_move_acting_reward, 0.0)),
    )
    new_reward_pl0 = acting_sign * acting_reward

    # If state was already done, keep state unchanged
    return jax.tree.map(
        lambda old, new: jnp.where(state.done, old, new),
        state,
        EnvState(
            board_type=next_board_type,
            board_owner=next_board_owner,
            board_visible=next_board_visible,
            board_moved=next_board_moved,
            board_piece_id=next_board_piece_id,
            init_arrangements=next_init_arrs,
            twosquare=next_twosquare,
            move_history=next_move_history,
            to_play=next_to_play,
            num_moves=next_num_moves,
            moves_since_attack=next_moves_since_attack,
            done=state.done | newly_done,
            reward_pl0=jnp.where(newly_done, new_reward_pl0, state.reward_pl0),
        ),
    )


def get_unknown_piece_info(state: EnvState):
    """Extract unknown opponent piece info for BeliefTransformer training & sampling.

    In Stratego, opponent pieces have piece_id in 0..39 (ordered by their starting cells).
    We sort unknown opponent pieces by their current board position (0..99 in acting POV)
    just like pystratego's `unknown_piece_*` tensors.

    Returns:
        unknown_pos_onehot: bool[40, 100]
        unknown_type_onehot: bool[40, 14]
        unknown_has_moved: bool[40]
        unknown_counts: int32[12] (remaining hidden opponent piece counts for types 0..11)
    """
    is_opp_hidden = (state.board_owner == 2) & (~state.board_visible)  # bool[100]

    # Count hidden opponent pieces per type (0..11)
    opp_hidden_types = jnp.where(is_opp_hidden, state.board_type, N_PIECE_TYPE)
    unknown_counts = jnp.bincount(opp_hidden_types, length=N_PLAYER_PIECE_TYPE).astype(jnp.int32)

    # Sort cells so that the K active unknown pieces come first in row-major order (0..99)
    # Give non-hidden cells key 100 + cell_idx so they sort to the end
    sort_keys = jnp.where(is_opp_hidden, jnp.arange(100, dtype=jnp.int32), 200 + jnp.arange(100, dtype=jnp.int32))
    sorted_cells = jnp.argsort(sort_keys)[:40]  # (40,)
    active_mask = is_opp_hidden[sorted_cells]   # (40,)

    pos_onehot = (jnp.arange(100)[None, :] == sorted_cells[:, None]) & active_mask[:, None]
    types = state.board_type[sorted_cells]
    type_onehot = (jnp.arange(N_PIECE_TYPE)[None, :] == types[:, None]) & active_mask[:, None]
    has_moved = state.board_moved[sorted_cells] & active_mask

    return pos_onehot, type_onehot, has_moved, unknown_counts


def compute_analytic_uniform_belief(
    board_owner: jax.Array,
    board_visible: jax.Array,
    board_moved: jax.Array,
    board_type: jax.Array,
) -> jax.Array:
    """Compute exact analytic uniform belief (100, 12) for opponent pieces (channels 12..23 in pystratego)."""
    is_opp = board_owner == 2
    is_opp_vis = is_opp & board_visible
    is_opp_hid = is_opp & (~board_visible)
    is_opp_hid_moved = is_opp_hid & board_moved
    is_opp_hid_unmoved = is_opp_hid & (~board_moved)

    hid_types = jnp.where(is_opp_hid, board_type, N_PIECE_TYPE)
    hid_counts = jnp.bincount(hid_types, length=N_PLAYER_PIECE_TYPE).astype(jnp.float32)  # (12,)

    n_hid_unmoved = jnp.sum(is_opp_hid_unmoved).astype(jnp.float32)

    # Immovable hidden pieces (Flag=10, Bomb=11) can ONLY be on unmoved hidden squares
    immovable_counts = hid_counts[10:12]  # (2,)
    prob_immovable_unmoved = immovable_counts / jnp.maximum(n_hid_unmoved, 1.0)  # (2,)

    # Remaining unmoved slots that hold movable pieces: n_hid_unmoved - sum(immovable_counts)
    movable_counts = hid_counts[:10]  # (10,)
    total_movable = jnp.maximum(jnp.sum(movable_counts), 1.0)
    movable_dist = movable_counts / total_movable  # (10,)

    frac_movable_on_unmoved = jnp.clip(
        (n_hid_unmoved - jnp.sum(immovable_counts)) / jnp.maximum(n_hid_unmoved, 1.0), 0.0, 1.0
    )

    # For unmoved hidden squares:
    unmoved_probs = jnp.concatenate([frac_movable_on_unmoved * movable_dist, prob_immovable_unmoved], axis=0)
    # For moved hidden squares: only movable types
    moved_probs = jnp.concatenate([movable_dist, jnp.zeros(2, dtype=jnp.float32)], axis=0)
    # Visible opponent squares: one-hot of true type
    vis_probs = jax.nn.one_hot(board_type, N_PLAYER_PIECE_TYPE, dtype=jnp.float32)

    probs = (
        is_opp_vis[:, None] * vis_probs
        + is_opp_hid_unmoved[:, None] * unmoved_probs[None, :]
        + is_opp_hid_moved[:, None] * moved_probs[None, :]
    )
    return probs  # (100, 12)


def observe(
    state: EnvState,
    max_num_moves: int = 2000,
    max_moves_between_attacks: int = 100,
) -> jax.Array:
    """Build infostate feature tensor of shape (92, C) for the 92 occupiable cells."""
    # 1. Own pieces one-hot: (100, 12)
    our_pieces = (state.board_owner == 1)[:, None] & (
        state.board_type[:, None] == jnp.arange(N_PLAYER_PIECE_TYPE)[None, :]
    )

    # 2. Opponent piece probabilities (analytic uniform belief + visible onehot): (100, 12)
    their_probs = compute_analytic_uniform_belief(
        state.board_owner, state.board_visible, state.board_moved, state.board_type
    )

    # 3. Opponent's belief over our pieces (symmetric from their perspective): (100, 12)
    swapped_owner = jnp.where(
        state.board_owner == 1, jnp.int8(2), jnp.where(state.board_owner == 2, jnp.int8(1), state.board_owner)
    )
    our_probs = compute_analytic_uniform_belief(
        swapped_owner, state.board_visible, state.board_moved, state.board_type
    )

    # 4. Basic state channels: (100, 7)
    our_hidden = ((state.board_owner == 1) & (~state.board_visible)).astype(jnp.float32)[:, None]
    their_hidden = ((state.board_owner == 2) & (~state.board_visible)).astype(jnp.float32)[:, None]
    empty_cell = (state.board_owner == 0).astype(jnp.float32)[:, None]
    our_moved = ((state.board_owner == 1) & state.board_moved).astype(jnp.float32)[:, None]
    their_moved = ((state.board_owner == 2) & state.board_moved).astype(jnp.float32)[:, None]
    moves_frac = jnp.full((100, 1), state.num_moves / float(max_num_moves), dtype=jnp.float32)
    attack_frac = jnp.full(
        (100, 1), state.moves_since_attack / float(max_moves_between_attacks), dtype=jnp.float32
    )

    # 5. Cemetery / dead pieces at starting squares: (100, 2)
    our_alive_ids = jnp.where(state.board_owner == 1, state.board_piece_id, -1)
    our_id_alive = jnp.zeros(40, dtype=jnp.bool_).at[jnp.clip(our_alive_ids, 0, 39)].set(our_alive_ids >= 0)
    our_dead_init = (~our_id_alive) & (state.init_arrangements[0] != EMPTY)
    our_dead_plane = jnp.zeros(100, dtype=jnp.float32).at[:40].set(our_dead_init.astype(jnp.float32))[:, None]

    their_alive_ids = jnp.where(state.board_owner == 2, state.board_piece_id, -1)
    their_id_alive = jnp.zeros(40, dtype=jnp.bool_).at[jnp.clip(their_alive_ids, 0, 39)].set(their_alive_ids >= 0)
    their_dead_init = (~their_id_alive) & (state.init_arrangements[1] != EMPTY)
    their_dead_plane = (
        jnp.zeros(100, dtype=jnp.float32).at[60:100].set(jnp.flip(their_dead_init).astype(jnp.float32))[:, None]
    )

    # 6. Move history planes: (100, MOVE_MEMORY)
    hist_planes = jnp.transpose(state.move_history, (1, 0))

    features_100 = jnp.concatenate(
        [
            our_pieces.astype(jnp.float32),
            their_probs,
            our_probs,
            our_hidden,
            their_hidden,
            empty_cell,
            our_moved,
            their_moved,
            moves_frac,
            attack_frac,
            our_dead_plane,
            their_dead_plane,
            hist_planes,
        ],
        axis=-1,
    )
    return features_100[OCCUPIABLE_INDICES]  # (92, C)


INFOSTATE_CHANNELS = 12 + 12 + 12 + 7 + 2 + MOVE_MEMORY  # 61 channels


def assign_opponent_hidden_pieces(state: EnvState, sampled_types: jax.Array) -> EnvState:
    """Replace hidden opponent piece types in `state` with `sampled_types` (int8[40] or onehot[40, 14]).

    Matches `pystratego.util.assign_opponent_hidden_pieces` used in SearchBot.
    """
    if sampled_types.ndim == 2:
        sampled_types = jnp.argmax(sampled_types, axis=-1).astype(jnp.int8)
    else:
        sampled_types = sampled_types.astype(jnp.int8)

    is_opp_hidden = (state.board_owner == 2) & (~state.board_visible)
    sort_keys = jnp.where(
        is_opp_hidden, jnp.arange(100, dtype=jnp.int32), 200 + jnp.arange(100, dtype=jnp.int32)
    )
    sorted_cells = jnp.argsort(sort_keys)[:40]
    active_mask = is_opp_hidden[sorted_cells]

    current_types = state.board_type[sorted_cells]
    updated_types = jnp.where(active_mask, sampled_types, current_types)
    new_board_type = state.board_type.at[sorted_cells].set(updated_types)
    return state._replace(board_type=new_board_type)
