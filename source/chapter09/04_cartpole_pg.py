import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ptan.agent import PolicyAgent, float32_preprocessor
from ptan.experience import ExperienceSourceFirstLast
from tensorboardX import SummaryWriter

from .lib import PGN, Batch


def train(
    gamma: float,
    beta: float,
    lr: float,
    count: int,
    batch_size: int,
    print_every_steps=20,
):
    env = gym.make("CartPole-v0")
    model = PGN(env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    agent = PolicyAgent(model, apply_softmax=True, preprocessor=float32_preprocessor)
    experience_source = ExperienceSourceFirstLast(env, agent, gamma, steps_count=count)
    writer = SummaryWriter(comment="-cartpole-pg")

    step_id = 0
    done_episodes = 0.0
    reward_acummulation = 0.0
    total_rewards = []
    batch = Batch()

    for step_id, experience in enumerate(experience_source):
        reward_acummulation += experience.reward
        baseline = reward_acummulation / (step_id + 1)
        batch.state.append(experience.state)
        batch.action.append(int(experience.action))
        batch.reward.append(experience.reward - baseline)
        writer.add_scalar("baseline", baseline, step_id)

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
                print(f"Solved in {step_id} steps {done_episodes:,} episodes!")
                break
        if len(batch.state) < batch_size:
            continue

        states = torch.FloatTensor(batch.state)
        action = torch.LongTensor(batch.action)
        scales = torch.FloatTensor(batch.reward)

        optimizer.zero_grad()
        logits = model(states)
        log_likelihood = F.log_softmax(logits, dim=1)
        Q = scales * log_likelihood[range(batch_size), action]
        policy_loss = -Q.mean()
        probability = F.softmax(logits, dim=1)
        entropy = (probability * log_likelihood).sum(dim=1).mean()
        entropy_loss = -beta * entropy
        loss = policy_loss + entropy_loss
        loss.backward()
        optimizer.step()

        # compute Kullback–Leibler divergence.
        last_prob = probability
        logits = model(states)
        probability = F.softmax(logits, dim=1)
        Kl = ((probability / last_prob).log() * last_prob).sum(dim=1).mean()
        writer.add_scalar("kl", Kl.item(), step_id)

        grad_max, grad_mean, grad_count = 0.0, 0.0, 0
        for param in model.parameters():
            grad_count += 1
            grad_max = max(grad_max, param.grad.abs().max().item())
            grad_mean += (param.grad ** 2).mean().sqrt().item()

        writer.add_scalar("baseline", baseline, step_id)
        writer.add_scalar("entropy", entropy.item(), step_id)
        writer.add_scalar("batch_scales", np.mean(batch.reward), step_id)
        writer.add_scalar("loss_entropy", entropy_loss.item(), step_id)
        writer.add_scalar("loss_policy", policy_loss.item(), step_id)
        writer.add_scalar("loss_total", loss.item(), step_id)
        writer.add_scalar("grad_l2", grad_mean / grad_count, step_id)
        writer.add_scalar("grad_max", grad_max, step_id)

        batch.state.clear()
        batch.action.clear()
        batch.reward.clear()

    writer.close()


if __name__ == "__main__":
    try:
        train(
            gamma=0.99, beta=0.01, lr=0.001, count=10, batch_size=8,
        )
    except KeyboardInterrupt:
        pass
