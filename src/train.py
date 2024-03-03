from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import numpy as np
from ppo import PPO
from stable_baselines3.common.monitor import Monitor
import time
from datetime import datetime

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self) -> None:
        self.agent = PPO(env)
    def act(self, observation, use_random=False):
        observation = torch.tensor(observation, dtype=torch.float32)
        # Normalize the observation
        observation = self.agent.normalize_obs(observation, training=False)
        return self.agent.policy(observation).argmax().item()

    def save(self, path):
        state_dict = {'policy_state_dict': self.agent.policy.state_dict(),
                      'optimizer_state_dict': self.agent.optimizer.state_dict(),
                      'observation_mean': self.agent.obs_running_mean,
                      'observation_var': self.agent.obs_running_var,
                      'reward_mean': self.agent.reward_running_mean,
                      'reward_var': self.agent.reward_running_var,}
        torch.save(state_dict, path)

        print(f"Model saved at {path}")
        print(f"Mean: {self.agent.obs_running_mean}")
        print(f"Var: {self.agent.obs_running_var}")


    def load(self):
        state_dict = torch.load("agent.pth")
        self.agent.policy.load_state_dict(state_dict['policy_state_dict'])
        self.agent.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.agent.obs_running_mean = state_dict['observation_mean']
        self.agent.obs_running_var = state_dict['observation_var']
        self.agent.reward_running_mean = state_dict['reward_mean']
        self.agent.reward_running_var = state_dict['reward_var']
        

    def train(self, total_timesteps=10_000, run_name="ppo"):
        self.agent.train(total_timesteps, 
                         make_env=lambda: Monitor(TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)),
                         run_name=run_name)

    def evaluate(self, env, nb_episode):
        rewards = []
        actions = [0] * 4
        for _ in range(nb_episode):
            s, _ = env.reset()
            done, truncated = False, False
            total_reward = 0
            while not done and not truncated:
                a = self.act(s)
                s, r, done, truncated, _ = env.step(a)
                total_reward += r
                actions[a] += 1
            rewards.append(total_reward)
        
        print(f"Actions: {actions / np.sum(actions) * 100}")
        return np.mean(rewards)


TRAIN = False

if __name__ == "__main__":
    agent = ProjectAgent()

    if TRAIN:
        current_time = datetime.now().strftime("%m%d%H%M%S")
        agent.train(total_timesteps=1_000_000, 
                    run_name=f"ppo_test_{current_time}")
        agent.save("agent.pth")
    else:
        agent.load()

    reward = agent.evaluate(env, 1)

    print(f"Reward: {reward}")
    print(f"Reward: {reward / 1e9 :.4f}e9")
