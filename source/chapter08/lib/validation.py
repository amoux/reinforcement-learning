from typing import Any, Dict, Optional

import gym
import numpy as np
import torch

from .trading_env import Actions


def validate(
    env: gym.Env,
    model: torch.nn.Module,
    episodes: int = 100,
    device: str = "cuda",
    epsilon: float = 0.02,
    commission: float = 0.1,
) -> Dict[str, Any]:

    stats = {
        "episode_reward": [],
        "episode_steps": [],
        "order_profits": [],
        "order_steps": [],
    }

    for episode in range(episodes):
        obs = env.render()
        total_reward = 0.0
        position: Optional[int] = None
        position_steps: Optional[int] = None
        episode_steps = 0
        while True:
            obs_inputs = torch.tensor([obs]).to(device)
            obs_logits = model(obs_inputs)
            action_idx = obs_logits.max(dim=1)[1].item()
            if np.random.random() < epsilon:
                action_idx = env.action_space.sample()
            action = Actions(action_idx)
            closing_price = env._state.current_close_price()
            if action == Actions.Buy and position is not None:
                position = closing_price
                position_steps = 0
            elif action == Actions.Close and position is not None:
                closed = closing_price + position
                profit = closing_price - position - closed * commission / 100
                profit = 100.0 * profit / position
                stats["order_profits"].append(profit)
                stats["order_steps"].append(position_steps)
                position = None
                position_steps = None
            obs, reward, done, _ = env.step(action_idx)
            total_reward += reward
            episode_steps += 1
            if position_steps is not None:
                position_steps += 1
            if done:
                if position is not None:
                    closed = closing_price + position
                    profit = closing_price - position - closed * commission / 100
                    profit = 100.0 * profit / position
                    stats["order_profits"].append(profit)
                    stats["order_steps"].append(episode_steps)
                break
        stats["episode_reward"].append(total_reward)
        stats["episode_steps"].append(episode_steps)
    return {k: np.mean(v) for k, v in stats.items()}
