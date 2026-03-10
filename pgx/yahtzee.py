# todo 

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

# Constants
NUM_CATEGORIES = 13
MAX_ROLLS = 2 # this only counts the rerolls therefore its only two. total rolls are 3
NUM_DICE = 5

# Category indices
ONES, TWOS, THREES, FOURS, FIVES, SIXES = 0, 1, 2, 3, 4, 5
THREE_OF_A_KIND, FOUR_OF_A_KIND, FULL_HOUSE = 6, 7, 8
SMALL_STRAIGHT, LARGE_STRAIGHT, YAHTZEE, CHANCE = 9, 10, 11, 12

@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros((NUM_DICE + NUM_CATEGORIES + 2,), dtype=jnp.float32)
    rewards: Array = jnp.zeros(2, dtype=jnp.float32)
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)
    legal_action_mask: Array = jnp.ones(32 + NUM_CATEGORIES, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)

    _dice: Array = jnp.zeros(NUM_DICE, dtype=jnp.int32)
    _rolls_left: Array = jnp.int32(MAX_ROLLS)
    _turn_phase: Array = jnp.int32(0)
    _categories_used: Array = jnp.zeros((2, NUM_CATEGORIES), dtype=jnp.bool_)
    _scores: Array = jnp.zeros((2, NUM_CATEGORIES), dtype=jnp.int32)
    _num_players: Array = jnp.int32(2)

    @property
    def env_id(self) -> core.EnvId:
        return "yahtzee"

class Yahtzee(core.Env):
    """
    Kniffel (Yahtzee) environment for 1-4 players.

    Action space:
    - Actions 0-31: Roll dice (binary mask of which dice to keep)
        - 0 = reroll all dice
        - 31 = keep all dice
    - Actions 32-44: Score in category:
        - 32 = ones
        - 44 = chance
    """

    def __init__(self, num_players: int = 2):
        super().__init__()

        assert 1 <= num_players <= 4, "Yahtzee supports 1-4 players"

        self._num_players = num_players

    def _init(self, key: PRNGKey) -> State:
        """
        Initialize game with first roll
        """
        
        # Roll initial dice
        dice = jax.random.randint(key, shape=(NUM_DICE,), minval=1, maxval=7)

        state = State(
            current_player=jnp.int32(0),
            _dice = dice,
            _rolls_left = jnp.int32(2),
            _turn_phase = jnp.int32(0),
            _categories_used = jnp.zeros((self._num_players, NUM_CATEGORIES), dtype = jnp.bool_),
            _scores = jnp.zeros((self._num_players, NUM_CATEGORIES), dtype=jnp.int32),
            _num_players = jnp.int32(self._num_players),
        )

        state = state.replace(
            legal_action_mask = _compute_legal_actions(state),
            observation = _make_observation(state, state.current_player),
        )

        return state
    
    def _step(self, state: core.State, action: Array, key) -> State:
        """
        Execute Action
        """

        assert isinstance(state, State)
        state = jax.lax.cond(
            action < 32,
            lambda: _handle_reroll(state, action, key),
            lambda: _handle_score(state, action - 32, key)
        )
        
        # Update legal actions
        state = state.replace(
            legal_action_mask = _compute_legal_actions(state)
        )

        # check if game is over
        all_categories_used = jnp.all(state._categories_used)
        state = state.replace(
            terminated = all_categories_used,
            rewards = jax.lax.select(
                all_categories_used,
                _compute_final_scores(state),
                jnp.zeros(self.num_players, dtype=jnp.float32),
            )
        )

        return state
        
    def _observe(self, state: core.State, player_id: Array) -> Array:
        """
            Generate observation for player
        """
        assert isinstance(state, State)
        return _make_observation(state, player_id)

    @property
    def id(self) -> core.EnvId:
        return "yahtzee"

    @property
    def version(self) -> str:
        return "v0"
    
    @property
    def num_players(self):
        return self._num_players
    
def _handle_reroll(state: State, action: Array, key: PRNGKey) -> State:
    """Reroll dice based on binary mask action"""

    keep_mask = jnp.array([
        (action >> i) & 1 for i in range(NUM_DICE)
    ], dtype=jnp.bool_)

    new_dice = jax.random.randint(key, shape=(NUM_DICE, ), minval=1, maxval=7)
    dice = jnp.where(keep_mask, state._dice, new_dice)

    # Decrease rolls left
    rolls_left = state._rolls_left - 1

    # if no rolls left
    turn_phase = jax.lax.select(
        rolls_left == 0,
        jnp.int32(1),
        jnp.int32(0)
    )

    return state.replace(
        _dice = dice,
        _rolls_left = rolls_left,
        _turn_phase = turn_phase
    )
    
def _handle_score(state: State, category: Array, key: PRNGKey) -> State:
    """Score current dice in chosen category"""
    player = state.current_player

    # Calculate score for this category
    score = _calculate_score(state._dice, category)

    scores = state._scores.at[player, category].set(score)
    categories_used = state._categories_used.at[player, category].set(True)

    # Move to next player
    next_player = (player + 1) % state._num_players

    # reset the roll and increase turn?

    return state.replace(
        current_player = next_player,
        _scores = scores,
        _categories_used = categories_used,
        _rolls_left = jnp.int32(MAX_ROLLS),
        _turn_phase = jnp.int32(0),
        _dice = jax.random.randint(key, (NUM_DICE,), minval=1, maxval=7),
    )

def _compute_legal_actions(state: State) -> Array:
    """
    Compute which actions are legal
    """

    legal = jnp.zeros(32 + NUM_CATEGORIES, dtype=jnp.bool_)

    # in rolling phase
    has_rolls = state._rolls_left > 0 

    # Can reroll (actions 0-31) if in rolling phase and have rolls left
    legal = legal.at[:32].set(has_rolls)

    # Can score (actions 32-44) if in scoreing phase and category not used
    scoring_phase = ~has_rolls
    player = state.current_player
    available_categories = ~state._categories_used[player]
    legal = legal.at[32:32+NUM_CATEGORIES].set(scoring_phase & available_categories)

    return legal

def _make_observation(state: State, player_id: Array) -> Array:
    """
    Create observation for player.

    Observation vector:
    - Dice values (5 floats, normalized 0-1)
    - Categories used by player (13)
    - Rolls left (1 float, normalized 0-1)
    - Is my turn (1 bool)

    These are the minimal requirements to give the observaiton the markov property
    """

    # Normalize dice to 0-1
    dice_normalized = (state._dice - 1) / 5.0

    # Get player's used categories
    categories_used = state._categories_used[player_id].astype(jnp.float32)


    # Rolls left
    rolls_normalized = state._rolls_left / float(MAX_ROLLS)

    # is_my_turn
    is_my_turn = (state.current_player == player_id).astype(jnp.float32)

    # todo add score card as well
    return jnp.concatenate([
        dice_normalized,                            # 5
        categories_used,                            # 13
        jnp.array([rolls_normalized, is_my_turn])   # 2
    ])                                              # 20

def _calculate_score(dice: Array, categorty: Array) -> Array:
    """
    Calculate score for given dice in given category
    """

    counts = jnp.array([
        jnp.sum(dice == i) for i in range(1, 7)
    ])

    # Upper section (ones through sixes)
    upper_scores = jnp.array([
        jnp.sum(jnp.where(dice == i, i, 0)) for i in range(1, 7)
    ])

    # Three of a kind
    three_of_kind = jax.lax.select(
        jnp.any(counts >= 3),
        jnp.sum(dice),
        jnp.int32(0)
    )

    # Four of a kind
    four_of_kind = jax.lax.select(
        jnp.any(counts >= 4),
        jnp.sum(dice),
        jnp.int32(0)
    )

    # Full house
    has_three = jnp.any(counts == 3)
    has_two = jnp.any(counts == 2)
    full_house = jax.lax.select(has_three & has_two, jnp.int32(25), jnp.int32(0))

    # Small Straight
    sorted_dice = jnp.sort(dice)
    unique_dice = jnp.unique(sorted_dice, size=NUM_DICE, fill_value=0)
    small_straight = jax.lax.select(
        _has_sequence(unique_dice, 4),
        jnp.int32(30),
        jnp.int32(0)
    )

    # Large straight
    large_straight = jax.lax.select(
        _has_sequence(unique_dice, 5),
        jnp.int32(40),
        jnp.int32(0)
    )

    # Yahtzee
    yahtzee = jax.lax.select(
        jnp.any(counts == 5),
        jnp.int32(50),
        jnp.int32(0)
    )

    # chance
    chance = jnp.sum(dice)

    all_scores = jnp.concatenate([
        upper_scores,
        jnp.array([
            three_of_kind,
            four_of_kind,
            full_house,
            small_straight,
            large_straight,
            yahtzee,
            chance
        ])
    ])

    return all_scores[categorty]

def _has_sequence(sorted_unique: Array, lenght: int) -> Array:
    """
    Check if sorted unique values contain a sequence of given length
    """
    # turn into onehot
    mask = jnp.zeros(7, dtype=jnp.int32).at[sorted_unique].set(1)

    mask = mask[1:]

    def count_consecutive(carry, is_present):
        current_run, max_run = carry

        current_run = jnp.where(is_present == 1, current_run + 1, 0)

        max_run = jnp.maximum(max_run, current_run)

        return (current_run, max_run), None
    
    (_, max_run), _ = jax.lax.scan(count_consecutive, (0,0), mask)

    return max_run >= lenght


def _compute_final_scores(state: State) -> Array:
    """
    Compute final scores with bonus
    """
    totals = jnp.sum(state._scores, axis=1)
    upper_sum = jnp.sum(state._scores[:, :6], axis=1)
    bonus = jnp.where(upper_sum >= 63, jnp.int32(35), jnp.int32(0))
    return (totals + bonus).astype(jnp.float32)