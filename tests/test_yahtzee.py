import jax
import random
import jax.numpy as jnp
from pgx.yahtzee import(
    State,
    Yahtzee,
    _handle_reroll,
    _handle_score,
    _compute_legal_actions,
    _make_observation,
    _calculate_score,
    _has_sequence,
    _compute_final_scores
)

seed = 1701
rng = jax.random.PRNGKey(seed)
env = Yahtzee
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)

_handle_reroll = jax.jit(_handle_reroll)
_handle_score = jax.jit(_handle_score)
_compute_legal_actions = jax.jit(_compute_legal_actions)
_make_observation = jax.jit(_make_observation)
_calculate_score = jax.jit(_calculate_score)
_has_sequence = jax.jit(_has_sequence)
_compute_final_scores = jax.jit(_compute_final_scores)

def make_test_state(
        current_player: jnp.ndarray,
        scores: jnp.ndarray,
        categories_played: jnp.ndarray,
        rolls_left: jnp.ndarray,
        dice: jnp.ndarray,
        legal_action_mask=jnp.zeros(32 + 13, dtype=jnp.bool_),
):
    return State(
        current_player=current_player,
        legal_action_mask=legal_action_mask,
        _dice=dice,
        _rolls_left=rolls_left,
        _categories_used=categories_played,
        _scores=scores,
    )


def test_init():
    env = Yahtzee()
    state = init(jax.random.PRNGKey(0))
    assert state.current_player == 0
    assert state._rolls_left == 2
    assert state._turn_phase == 0
    assert jnp.all(state._dice >= 1) and jnp.all(state._dice <= 6)
    assert jnp.sum(state._categories_used) == 0


def test_compute_legal_actions_rolling_phase():
    state = make_test_state(
        current_player=jnp.int32(0),
        scores=jnp.zeros((2, 13), dtype=jnp.int32),
        categories_played=jnp.zeros((2, 13), dtype=jnp.bool_),
        rolls_left=jnp.int32(2),
        dice=jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32),
    )
    legal_mask = _compute_legal_actions(state)
    assert jnp.sum(legal_mask[:32]) > 0
    assert jnp.sum(legal_mask[32:]) == 0


def test_compute_legal_actions_scoring_phase():
    categories_used = jnp.zeros((2, 13), dtype=jnp.bool_)
    state = make_test_state(
        current_player=jnp.int32(0),
        scores=jnp.zeros((2, 13), dtype=jnp.int32),
        categories_played=categories_used,
        rolls_left=jnp.int32(0),
        dice=jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32),
    )
    state = state.replace(_turn_phase=jnp.int32(1))
    legal_mask = _compute_legal_actions(state)
    assert jnp.sum(legal_mask[:32]) == 0
    assert jnp.sum(legal_mask[32:]) == 13


def test_handle_reroll():
    for _ in range(100):
        dice = jnp.array([random.randint(1, 6) for _ in range(5)], dtype=jnp.int32)
        action = random.randint(0, 31)  # Random action (0-31 for dice selection)
        state = make_test_state(
        current_player=jnp.int32(0),
        scores=jnp.zeros((2, 13), dtype=jnp.int32),
        categories_played=jnp.zeros((2, 13), dtype=jnp.bool_),
        rolls_left=jnp.int32(2),
        dice=dice,
        )
        new_state = _handle_reroll(state, action, jax.random.PRNGKey(random.randint(0, 10000)))
        assert new_state._rolls_left == 1
        assert new_state._turn_phase == 0
        # Verify kept dice match original
        for i in range(5):
            if action & (1 << i):
                assert new_state._dice[i] == dice[i]


def test_handle_reroll_no_rolls_left():
    dice = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    state = make_test_state(
        current_player=jnp.int32(0),
        scores=jnp.zeros((2, 13), dtype=jnp.int32),
        categories_played=jnp.zeros((2, 13), dtype=jnp.bool_),
        rolls_left=jnp.int32(1),
        dice=dice,
    )
    new_state = _handle_reroll(state, 0, jax.random.PRNGKey(0))
    assert new_state._rolls_left == 0
    assert new_state._turn_phase == 1


def test_handle_score():
    scores = jnp.zeros((2, 13), dtype=jnp.int32)
    categories_used = jnp.zeros((2, 13), dtype=jnp.bool_)
    dice = jnp.array([1, 1, 2, 3, 4], dtype=jnp.int32)
    state = make_test_state(
        current_player=jnp.int32(0),
        scores=scores,
        categories_played=categories_used,
        rolls_left=jnp.int32(0),
        dice=dice,
    )
    state = state.replace(_turn_phase=jnp.int32(1))
    new_state = _handle_score(state, jnp.int32(0), jax.random.PRNGKey(0))  # Score ones
    assert new_state.current_player == 1
    assert new_state._categories_used[0, 0] == True
    assert new_state._rolls_left == 2
    assert new_state._turn_phase == 0


def test_calculate_score_upper_section():
    dice = jnp.array([1, 1, 2, 3, 4], dtype=jnp.int32)
    score_ones = _calculate_score(dice, jnp.int32(0))
    assert score_ones == 2
    
    score_twos = _calculate_score(dice, jnp.int32(1))
    assert score_twos == 2


def test_calculate_score_three_of_a_kind():
    dice = jnp.array([2, 2, 2, 4, 5], dtype=jnp.int32)
    score = _calculate_score(dice, jnp.int32(6))  # THREE_OF_A_KIND
    assert score == 15


def test_calculate_score_full_house():
    dice = jnp.array([2, 2, 3, 3, 3], dtype=jnp.int32)
    score = _calculate_score(dice, jnp.int32(8))  # FULL_HOUSE
    assert score == 25


def test_calculate_score_small_straight():
    dice = jnp.array([1, 2, 3, 4, 6], dtype=jnp.int32)
    score = _calculate_score(dice, jnp.int32(9))  # SMALL_STRAIGHT
    assert score == 30


def test_calculate_score_large_straight():
    dice = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    score = _calculate_score(dice, jnp.int32(10))  # LARGE_STRAIGHT
    assert score == 40


def test_calculate_score_yahtzee():
    dice = jnp.array([3, 3, 3, 3, 3], dtype=jnp.int32)
    score = _calculate_score(dice, jnp.int32(11))  # YAHTZEE
    assert score == 50


def test_calculate_score_chance():
    dice = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    score = _calculate_score(dice, jnp.int32(12))  # CHANCE
    assert score == 15


def test_has_sequence():
    unique = jnp.array([1, 2, 3, 4, 0], dtype=jnp.int32)
    result = _has_sequence(unique, 4)
    assert result == True
    
    result = _has_sequence(unique, 5)
    assert result == True


def test_make_observation():
    dice = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    categories_used = jnp.zeros((2, 13), dtype=jnp.bool_)
    state = make_test_state(
        current_player=jnp.int32(0),
        scores=jnp.zeros((2, 13), dtype=jnp.int32),
        categories_played=categories_used,
        rolls_left=jnp.int32(2),
        dice=dice,
    )
    obs = _make_observation(state, jnp.int32(0))
    assert obs.shape == (20,)
    assert obs[18] == 1.0  # is_my_turn


def test_compute_final_scores():
    scores = jnp.array([[20, 15, 10, 5, 8, 12, 0, 0, 25, 30, 0, 0, 15], 
                            [15, 12, 10, 8, 6, 9, 0, 0, 25, 30, 40, 0, 12]], dtype=jnp.int32)
    state = make_test_state(
        current_player=jnp.int32(0),
        scores=scores,
        categories_played=jnp.ones((2, 13), dtype=jnp.bool_),
        rolls_left=jnp.int32(0),
        dice=jnp.array([1, 1, 1, 1, 1], dtype=jnp.int32),
    )
    final_scores = _compute_final_scores(state)
    upper_sum_p0 = 20 + 15 + 10 + 5 + 8 + 12
    expected_p0 = upper_sum_p0 + 35 + 140  # with bonus
    assert final_scores[0] == expected_p0
