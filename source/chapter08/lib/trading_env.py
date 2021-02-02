import enum
from typing import Any, Tuple

import gym
import gym.spaces
import numpy as np
from gym.utils import seeding

from . import data_utils
from .data_utils import Prices

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    def __init__(
        self,
        bars: int,
        volume: int,
        commission: float,
        do_reset_on_close: bool,
        do_reward_on_close: bool,
    ) -> None:

        self.bars = bars
        self.volume = volume
        self.commission = commission
        self.do_reset_on_close = do_reset_on_close
        self.do_reward_on_close = do_reward_on_close
        self.maybe_has_position = False
        self.current_open_price = 0.0

    def reset(self, prices: Prices, offset: int) -> None:
        assert isinstance(prices, Prices)
        assert offset >= self.bars - 1
        self.maybe_has_position = False
        self.current_open_price = 0.0
        self._prices = prices
        self._offset = offset

    @property
    def shape(self) -> Tuple[int, ...]:
        # [h, l, c] * bars + position flag + rel_profit (since open)
        if self.volume:
            return (4 * self.bars + 1 + 1,)
        else:
            return (3 * self.bars + 1 + 1,)

    def encode(self) -> np.ndarray:
        """Convert the current state input into numpy array type."""
        state = np.ndarray(shape=self.shape, dtype=np.float32)
        shift_index: int = 0
        for bar in range(-self.bars + 1, 1):
            state[shift_index] = self._prices.high[self._offset + bar]
            shift_index += 1
            state[shift_index] = self._prices.low[self._offset + bar]
            shift_index += 1
            state[shift_index] = self._prices.close[self._offset + bar]
            shift_index += 1
            if self.volume:
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

    def current_close_price(self) -> float:
        """Compute the real closing price for the current bar."""
        opening = self._prices.open[self._offset]
        closing = self._prices.close[self._offset]
        real_price = opening * (1.0 + closing)
        return real_price

    def step(self, action: Actions) -> Tuple[float, Any]:
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
            done != self.do_reset_on_close
            if self.do_reward_on_close:
                reward += (
                    100.0 * (close - self.current_open_price) / self.current_open_price
                )
            self.maybe_has_position = False
            self.current_open_price = 0.0
        self._offset += 1
        prev_close = close
        close = self.current_close_price()
        done |= self._offset >= self._prices.close.shape[0] - 1
        if self.maybe_has_position and not self.do_reward_on_close:
            reward += 100.0 * (close - prev_close) / prev_close
        return reward, done

