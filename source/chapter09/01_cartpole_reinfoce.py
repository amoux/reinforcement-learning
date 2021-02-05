from typing import List

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ptan.agent import PolicyAgent, float32_preprocessor
from ptan.experience import ExperienceSourceFirstLast
from tensorboardX import SummaryWriter

from .lib import PGN, Batch


def compute_q_values(rewards: List[float], gamma: float) -> List[float]:
    output = []
    accumulation = 0.0
    for reward in reversed(rewards):
        accumulation *= gamma
        accumulation += reward
        output.append(accumulation)
    return list(reversed(output))


def train(gamma: float, lr: float, num_train_episodes: int, print_every_steps=20):

    env = gym.make("CartPole-v0")
    model = PGN(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    agent = PolicyAgent(model, apply_softmax=True, preprocessor=float32_preprocessor)
    experience_source = ExperienceSourceFirstLast(env, agent, gamma=gamma)
    writer = SummaryWriter(comment="cartpole-reinfoce")

    step_id = 0
    done_episodes = 0
    batch_episodes = 0
    total_rewards = []
    batch = Batch()

    for step_id, experience in enumerate(experience_source):
        batch.action.append(int(experience.action))
        batch.state.append(experience.state)
        batch.reward.append(experience.reward)

        if experience.last_state is None:
            current_reward = compute_q_values(batch.reward, gamma=gamma)
            batch.q_vals.extend(current_reward)
            batch.reward.clear()
            batch_episodes += 1

        obtained_rewards = experience_source.pop_total_rewards()
        if obtained_rewards:
            done_episodes += 1
            reward = obtained_rewards[0]
            total_rewards.append(reward)
            mean_rewards = float(np.mean(total_rewards[-100:]))
            if step_id % print_every_steps == 0:
                print(
                    "{}:\treward: {:6.2f}\tmean_100: {:6.2f}\tepisode: {:6.2f}".format(
                        step_id, reward, mean_rewards, done_episodes
                    )
                )
            writer.add_scalar("reward", reward, step_id)
            writer.add_scalar("reward_100", mean_rewards, step_id)
            writer.add_scalar("episode", done_episodes, step_id)
            if mean_rewards > 195:
                print(f"Solved in {step_id} steps and {done_episodes} episodes!")
                break
        if batch_episodes < num_train_episodes:
            continue

        action = torch.LongTensor(batch.action)
        states = torch.FloatTensor(batch.state)
        q_vals = torch.FloatTensor(batch.q_vals)

        optimizer.zero_grad()
        logits = model(states)
        log_likelihood = F.log_softmax(logits, dim=1)
        Q = q_vals * log_likelihood[range(len(states)), action]
        loss = -Q.mean()
        loss.backward()
        optimizer.step()

        batch_episodes = 0
        batch.state.clear()
        batch.action.clear()
        batch.q_vals.clear()

    writer.close()


if __name__ == "__main__":
    try:
        train(gamma=0.99, lr=0.01, num_train_episodes=4)
    except KeyboardInterrupt:
        pass
