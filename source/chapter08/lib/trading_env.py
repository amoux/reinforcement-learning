import enum
from typing import Any, Dict, List, Optional, Tuple

import gym
import gym.spaces
import numpy as np
from gym.utils import seeding

from . import data_utils
from .data_utils import Prices


def load_prices_from_files(data_dir: str) -> Dict[str, Prices]:
    prices = {
        file: data_utils.load_relative(file)
        for file in data_utils.price_files(data_dir)
    }
    return prices


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class InternalState:
    __slots__ = (
        "bars",
        "commission",
        "do_reset_on_close",
        "do_reward_on_close",
        "do_switch_on_volumes",
        "maybe_has_position",
        "current_open_price",
        "_prices",
        "_offset",
    )

    def __init__(
        self,
        bars: int = 10,
        commission: float = 0.1,
        do_reset_on_close: bool = True,
        do_reward_on_close: bool = False,
        do_switch_on_volumes: bool = True,
    ) -> None:
        """Internal representation of the environment's functionality.

        :param bars: number of bars to include in the observation space.
        :param commission: percentage of the stock price that the agent
            needs to "pay" to the broker on buying and selling stock.
        :param do_reset_on_close: TODO: <Finish Adding Documentation>
        :param do_reward_on_close: switch between the two reward schemes. If
            true, the agent will receive a reward only on the `close` action.
            Otherwise, it uses a small reward every bar; corresponding to price
            movement during that bar.
        :param do_switch_on_volumes: switch on volumes in observations.
        """
        assert isinstance(bars, int)
        assert isinstance(commission, float)
        assert bars > 0
        assert commission >= 0.0
        self.bars = bars
        self.commission = commission
        self.do_reset_on_close = do_reset_on_close
        self.do_reward_on_close = do_reward_on_close
        self.do_switch_on_volumes = do_switch_on_volumes
        self.maybe_has_position = False
        self.current_open_price = 0.0
        self._prices: Optional[Prices] = None
        self._offset: Optional[int] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def encode_self(self) -> np.ndarray:
        """Encoding including prices, with optional volumes and two numbers
        indicating the presence of a bought share and position profit."""
        raise NotImplementedError

    def current_close_price(self) -> float:
        """Calculate the current bar's close price.

        Prices passed to the `InternalState` have the relative ratios to the open
        price. This representation "can" help the agent learn price patterns that
        are independent of actual price value.
        """
        opening = self._prices.open[self._offset]
        closing = self._prices.close[self._offset]
        real_price = opening * (1.0 + closing)
        return real_price

    def reset(self, prices: Prices, offset: int) -> None:
        assert isinstance(prices, Prices)
        assert offset >= self.bars - 1
        self.maybe_has_position = False
        self.current_open_price = 0.0
        self._prices = prices
        self._offset = offset

    def step(self, action: Actions) -> Tuple[float, Any]:
        """Compute the reward in a percentage and indication of the episode ending.

        If the agent decides to buy a share, the state takes the latter to pay the
        commission. Usually, the agent can execute an order on a different price
        called; `price slippage.` If we have a position and the agent's request is
        to `close,` then; the commission is paid again, done flag updated, and if
        the `do_reset_on_close` is `True` - collect the final reward for the whole
        position and update the state.
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self.current_close_price()
        if action == Actions.Buy and not self.maybe_has_position:
            self.maybe_has_position = True
            self.current_open_price = close
            reward -= self.commission
        elif action == Actions.Close and self.maybe_has_position:
            reward -= self.commission
            done |= self.do_reset_on_close
            if self.do_reward_on_close:
                reward += (
                    100.0 * (close - self.current_open_price) / self.current_open_price
                )
            self.maybe_has_position = False
            self.current_open_price = 0.0
        # In the following, we modify the current offset
        # and give the reward for the last bar movement.
        self._offset += 1
        prev_close = close
        close = self.current_close_price()
        done |= self._offset >= self._prices.close.shape[0] - 1
        if self.maybe_has_position and not self.do_reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close
        return reward, done


class VectorDataState(InternalState):
    def __init__(self, *args, **kwargs) -> None:
        super(VectorDataState, self).__init__(*args, **kwargs)

    @property
    def shape(self) -> Tuple[int, ...]:
        # [h, l, c] * bars + position-flag + true-profit (since open)
        if self.do_switch_on_volumes:
            return (4 * self.bars + 1 + 1,)
        else:
            return (3 * self.bars + 1 + 1,)

    def encode_self(self) -> np.ndarray:
        """Convert and shift the current self state into numpy array type."""
        state = np.ndarray(shape=self.shape, dtype=np.float32)
        shift_index: int = 0
        for bar in range(-self.bars + 1, 1):
            state[shift_index] = self._prices.high[self._offset + bar]
            shift_index += 1
            state[shift_index] = self._prices.low[self._offset + bar]
            shift_index += 1
            state[shift_index] = self._prices.close[self._offset + bar]
            shift_index += 1
            if self.do_switch_on_volumes:
                state[shift_index] = self._prices.volume[self._offset + bar]
                shift_index += 1
        state[shift_index] = float(self.maybe_has_position)
        shift_index += 1
        if not self.maybe_has_position:
            state[shift_index] = 0.0
        else:
            price_diff = self.current_close_price() - self.current_open_price
            state[shift_index] = price_diff / self.current_open_price
        return state


class MatrixDataState(InternalState):
    def __init__(self, *args, **kwargs) -> None:
        super(MatrixDataState, self).__init__(*args, **kwargs)

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.do_switch_on_volumes:
            return (6, self.bars)
        else:
            return (5, self.bars)

    def encode_self(self) -> np.ndarray:
        """Encode the prices in the state matrix.

        - Based on the subsequent conditions
          - Depending on the current offset
          - Whether the agent needs volumes
          - Whether the agent has free stock
        """
        state = np.zeros(shape=self.shape, dtype=np.float32)
        bars = self.bars - 1
        state[0] = self._prices.high[self._offset - bars : self._offset + 1]
        state[1] = self._prices.low[self._offset - bars : self._offset + 1]
        state[2] = self._prices.close[self._offset - bars : self._offset + 1]
        if self.do_switch_on_volumes:
            state[3] = self._prices.volume[self._offset - bars : self._offset + 1]
            distance = 4
        else:
            distance = 3
        if self.maybe_has_position:
            state[distance] = 1.0
            price_diff = self.current_close_price() - self.current_open_price
            state[distance + 1] = price_diff / self.current_open_price
        return state


class StockEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        prices: Dict[str, Prices],
        state_type: str = "vector",
        bars: int = 10,
        commission: float = 0.1,
        do_reset_on_close: bool = True,
        do_reward_on_close: bool = False,
        do_switch_on_volumes: bool = False,
        do_random_offsets_on_reset: bool = True,
    ) -> None:
        self._prices = prices
        if state_type == "matrix":
            self._state = MatrixDataState(
                bars=bars,
                commission=commission,
                do_reset_on_close=do_reset_on_close,
                do_reward_on_close=do_reward_on_close,
                do_switch_on_volumes=do_switch_on_volumes,
            )
        else:
            self._state = VectorDataState(
                bars=bars,
                commission=commission,
                do_reset_on_close=do_reset_on_close,
                do_reward_on_close=do_reward_on_close,
                do_switch_on_volumes=do_switch_on_volumes,
            )
        self.do_random_offsets_on_reset = do_random_offsets_on_reset
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(
            low=np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32,
        )
        self.np_random: Optional[np.ndarray] = None
        self.seed()

    def close(self) -> None:
        """Called on the environment's destruction to allocate resources."""
        pass

    def render(self, mode="human", close=False) -> None:
        pass

    def seed(self, seed=None) -> List[np.ndarray]:
        """Gym's random number generator (adaptable to multiple instances)."""
        self.np_random, seed_0 = seeding.np_random(seed)
        seed_1 = seeding.hash_seed(seed_0 + 1) % 2 ** 31
        return [seed_0, seed_1]

    def reset(self) -> np.ndarray:
        """Make selection of the instrument and it's offset reseting the state.

        This method has to handle the action chosen by the agent and return
        the next `observation, reward, and done` flag.
        """
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars
        if self.do_random_offsets_on_reset:
            offset = self.np_random.choice(prices.high.shape[0] - bars * 10) + bars
        else:
            offset = bars
        self._state.reset(prices=prices, offset=offset)
        self_encoded_state = self._state.encode_self()
        return self_encoded_state

    def step(self, action_id: int) -> Tuple[np.ndarray, ...]:
        action = Actions(action_id)
        reward, done = self._state.step(action)
        self_encoded_observation = self._state.encode_self()
        info = {"instrument": self._instrument, "offset": self._state._offset}
        return self_encoded_observation, reward, done, info

    @classmethod
    def from_dir(cls, data_dir: str, **kwargs) -> "StockEnv":
        prices = load_prices_from_files(data_dir=data_dir)
        return StockEnv(prices, **kwargs)
