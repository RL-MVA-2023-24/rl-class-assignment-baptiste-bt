import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataclasses import dataclass
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
import time

@dataclass
class Args:
    learning_rate: float = 3e-4
    horizon: int = 256
    batch_size: int = 64
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    clip_coef_vf: float = 10.
    normalize_advantage: bool = True
    normalize_return: bool = True
    normalize_obs: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    num_envs: int = 12
    seed: int = 42
    device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActorCriticPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.actor(x)
    
    def get_value(self, x):
        return self.critic(x)
    
    def get_action_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class PPO:
    def __init__(self, env: gym.Env) -> None:
        self.args = Args()
        self.policy = ActorCriticPolicy(env.observation_space.shape[0], env.action_space.n, self.args.hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.args.learning_rate)
        self.env = env
        if self.args.normalize_obs:
            self.obs_running_mean = None
            self.obs_running_var = None
            self.reward_running_mean = None
            self.reward_running_var = None
    #@profile
    def train(self, total_timesteps, make_env, run_name):
        # Initialization
        envs = gym.vector.AsyncVectorEnv([make_env for _ in range(self.args.num_envs)])
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
            )
        device = self.args.device
        epsilon = self.args.clip_coef

        rollout_buffer = RolloutBuffer(self.args.horizon, self.args.num_envs, device, self.env.observation_space.shape)
        
        iterations = total_timesteps // (self.args.horizon * self.args.num_envs) + 1
        start_time = time.time()
        next_obs, _ = envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs, device=device)
        if self.args.normalize_obs:
            next_obs = self.normalize_obs(next_obs)
        
        next_done = torch.zeros(self.args.num_envs, device=device)

        total_steps = 0
        total_episodes = 0
        rolling_rewards = np.zeros(10, dtype=np.float32)
        best_mean_reward = -np.inf

        progress_bar = tqdm(range(1, iterations + 1))
        for i in progress_bar:
            progress_bar.set_postfix_str(f"Episodic return: {rolling_rewards.mean() / 1e9:.4f}")

            # Collect rollouts
            for step in range(self.args.horizon):
                total_steps += self.args.num_envs

                rollout_buffer.obs[step] = next_obs
                rollout_buffer.dones[step] = next_done

                with torch.no_grad():
                    action, log_prob, entropy, value = self.policy.get_action_value(next_obs)
                    rollout_buffer.values[step] = value.flatten()
                    rollout_buffer.actions[step] = action
                    rollout_buffer.logprobs[step] = log_prob

                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)

                if self.args.normalize_return:
                    rollout_buffer.rewards[step] = self.normalize_rewards(torch.tensor(reward, device=device).flatten())
                else:
                    rollout_buffer.rewards[step] = torch.tensor(reward, device=device).flatten()
                next_obs, next_done = torch.Tensor(next_obs, device=device), torch.Tensor(next_done, device=device)
                if self.args.normalize_obs:
                    next_obs = self.normalize_obs(next_obs)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            rolling_rewards[total_episodes % 10] = info["episode"]["r"]
                            total_episodes += 1
                            rolling_mean = rolling_rewards.mean()
                            writer.add_scalar("charts/episodic_return", rolling_mean, total_steps)
                            # writer.add_scalar("charts/episodic_return", info["episode"]["r"], total_steps)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], total_steps)

                            if rolling_mean > best_mean_reward:
                                best_mean_reward = rolling_mean
                                self.save(f"models/{run_name}_best.pth")
                                print(f"Best model saved at {total_steps} steps with reward {best_mean_reward}")

            # Bootstrap value if not done
            # Inspired from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
            with torch.no_grad():
                next_value = self.policy.get_value(next_obs).reshape(1, -1)
                lastgaelam = 0
                for t in reversed(range(self.args.horizon)):
                    if t == self.args.horizon - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - rollout_buffer.dones[t + 1]
                        nextvalues = rollout_buffer.values[t + 1]
                    delta = rollout_buffer.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - rollout_buffer.values[t]
                    rollout_buffer.advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                rollout_buffer.returns = rollout_buffer.advantages + rollout_buffer.values
                
            dataset = TensorDataset(rollout_buffer.obs.reshape(-1, *self.env.observation_space.shape), 
                                    rollout_buffer.actions.reshape(-1),
                                    rollout_buffer.logprobs.reshape(-1), 
                                    rollout_buffer.advantages.reshape(-1), 
                                    rollout_buffer.values.reshape(-1), 
                                    rollout_buffer.returns.reshape(-1))
            dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

            clip_fraction = []
            # Training
            # Inspired from https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
            for epoch in range(self.args.n_epochs):
                for obs, actions, old_logprobs, advantages, old_values, returns in dataloader:
                    _, next_logprobs, entropy, next_values = self.policy.get_action_value(obs, actions)

                    log_ratio = next_logprobs - old_logprobs
                    ratio = torch.exp(log_ratio)

                    # Compute appoximate KL
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_fraction += [((ratio - 1.0).abs() > epsilon).float().mean().item()]

                    # Normalize advantage
                    if self.args.normalize_advantage:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages).mean()

                    if self.args.clip_vloss:
                        value_pred_clipped = old_values + (next_values - old_values).clamp(-self.args.clip_coef_vf, self.args.clip_coef_vf)
                        value_loss = 0.5 * torch.max((returns - value_pred_clipped) ** 2, (returns - next_values) ** 2).mean()
                    else:
                        value_loss = 0.5 * F.mse_loss(returns, next_values)

                    entropy_loss = -entropy.mean()

                    loss = policy_loss + self.args.vf_coef * value_loss + self.args.ent_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()


            y_pred, y_true = rollout_buffer.values.reshape(-1).cpu().numpy(), rollout_buffer.returns.reshape(-1).cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], total_steps)
            writer.add_scalar("losses/value_loss", value_loss.item(), total_steps)
            writer.add_scalar("losses/policy_loss", policy_loss.item(), total_steps)
            writer.add_scalar("losses/entropy", entropy_loss.item(), total_steps)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), total_steps)
            writer.add_scalar("losses/clipfrac", np.mean(clip_fraction), total_steps)
            writer.add_scalar("losses/explained_variance", explained_var, total_steps)
            writer.add_scalar("charts/SPS", int(total_steps / (time.time() - start_time)), total_steps)
        
            # Clean the buffer
            rollout_buffer.clean()

        envs.close()
        writer.close()
    
    def normalize_obs(self, obs, training=True):
        if training:
            if self.obs_running_mean is None:
                self.obs_running_mean = obs.mean(dim=0)
                self.obs_running_var = obs.var(dim=0)
            else:
                alpha = 0.99
                # Compute running statistics
                self.obs_running_mean = alpha * self.obs_running_mean + (1 - alpha) * obs.mean(dim=0)
                self.obs_running_var = alpha * self.obs_running_var + (1 - alpha) * obs.var(dim=0)

        # Normalize the observations
        obs = (obs - self.obs_running_mean) / (torch.sqrt(self.obs_running_var) + 1e-8)
        return obs
    
    def normalize_rewards(self, rewards, training=True):
        if training:
            if self.reward_running_mean is None:
                self.reward_running_mean = rewards.mean()
                self.reward_running_var = rewards.var()
            else:
                alpha = 0.99
                # Compute running statistics
                self.reward_running_mean = alpha * self.reward_running_mean + (1 - alpha) * rewards.mean()
                self.reward_running_var = alpha * self.reward_running_var + (1 - alpha) * rewards.var()

        # Normalize the rewards
        rewards = (rewards - self.reward_running_mean) / (torch.sqrt(self.reward_running_var) + 1e-8)
        return rewards
    
    def save(self, path):
        state_dict = {'policy_state_dict': self.policy.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'obs_running_mean': self.obs_running_mean,
                      'obs_running_var': self.obs_running_var,
                      'reward_running_mean': self.reward_running_mean,
                      'reward_running_var': self.reward_running_var,}
        torch.save(state_dict, path)
        print(f"Model saved at {path}")
    
        
class RolloutBuffer:
    def __init__(self, horizon, num_envs, device, obs_shape) -> None:
        self.obs = torch.zeros((horizon, num_envs) + obs_shape, device=device)
        self.actions = torch.zeros((horizon, num_envs), device=device)
        self.logprobs = torch.zeros((horizon, num_envs), device=device)
        self.rewards = torch.zeros((horizon, num_envs), device=device)
        self.dones = torch.zeros((horizon, num_envs), device=device)
        self.values = torch.zeros((horizon, num_envs), device=device)
        self.advantages = torch.zeros((horizon, num_envs), device=device)
        self.returns = torch.zeros((horizon, num_envs), device=device)

    def clean(self):
        self.obs = self.obs * 0
        self.actions = self.actions * 0
        self.logprobs = self.logprobs * 0
        self.rewards = self.rewards * 0
        self.dones = self.dones * 0
        self.values = self.values * 0
        self.advantages = self.advantages * 0
        self.returns = self.returns * 0

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = PPO(env)
    agent.train(100_000, lambda: Monitor(gym.make("CartPole-v1")), f"ppo_cartpole_{int(time.time())}")