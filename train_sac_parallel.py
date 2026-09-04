"""
Parallelized SAC training for CarRacing-v2.

Uses gymnasium's AsyncVectorEnv to run several CarRacing instances in separate
OS processes, so that Box2D physics + rendering (the real bottleneck, since
this env's observation IS a rendered frame) happens concurrently across CPU
cores instead of one step at a time in a single process.

IMPORTANT (Windows): multiprocessing here uses the 'spawn' start method, which
re-imports this file in each worker process. Everything that should run only
once, in the main process, MUST live inside `if __name__ == "__main__":`.
Do NOT run this inside a Jupyter notebook on Windows -- a notebook can't be
re-imported as a module, so worker processes won't be able to find the
class/function definitions they need. Run it as a plain script:

    py -3 train_sac_parallel.py
"""

import json
import os
import time

# Windows/conda: torch (MKL) and opencv-python each bundle their own Intel
# OpenMP runtime (libiomp5md.dll); loading both in one process trips
# "OMP: Error #15" and kills the process. Must be set before those libraries
# are imported. AsyncVectorEnv's 'spawn' workers re-execute this module top
# to bottom, so this also protects each worker process, not just main.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import gymnasium as gym
import matplotlib

matplotlib.use('Agg')  # headless: only ever save figures, never try to pop up a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.vector import AsyncVectorEnv


# ==================== Preprocessing (same as the notebook) ====================

def preprocess(img):
    img = img[:84, 6:90]  # CarRacing-v2-specific cropping
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return img


class ImageEnv(gym.Wrapper):
    def __init__(self, env, skip_frames=4, stack_frames=4, initial_no_op=50, **kwargs):
        super(ImageEnv, self).__init__(env, **kwargs)
        self.initial_no_op = initial_no_op
        self.skip_frames = skip_frames
        self.stack_frames = stack_frames

        if isinstance(env.action_space, gym.spaces.Box):
            self.no_op_action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            self.no_op_action = 0

        # ImageEnv changes the observation's actual shape/dtype vs. the wrapped
        # env's raw observation_space. AsyncVectorEnv's shared-memory buffers are
        # sized from this attribute, so it MUST be kept in sync (the notebook's
        # single-env version never needed this since nothing read the attribute).
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(stack_frames, 84, 84), dtype=np.float32
        )

    def reset(self, **kwargs):
        s, info = self.env.reset(**kwargs)
        for i in range(self.initial_no_op):
            s, r, terminated, truncated, info = self.env.step(self.no_op_action)
        s = preprocess(s)
        self.stacked_state = np.tile(s, (self.stack_frames, 1, 1))  # [4, 84, 84]
        return self.stacked_state, info

    def step(self, action):
        reward = 0
        for _ in range(self.skip_frames):
            s, r, terminated, truncated, info = self.env.step(action)
            reward += r
            if terminated or truncated:
                break
        s = preprocess(s)
        self.stacked_state = np.concatenate((self.stacked_state[1:], s[np.newaxis]), axis=0)
        return self.stacked_state, reward, terminated, truncated, info


def make_env():
    """Top-level (picklable) env constructor -- required by AsyncVectorEnv workers."""
    env = gym.make('CarRacing-v2', continuous=True, render_mode='rgb_array')
    return ImageEnv(env)


# ==================== Networks (same architecture as the notebook) ====================

LOG_STD_MIN, LOG_STD_MAX = -20, 2
ACTION_LOW = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def rescale_action(tanh_action):
    """Map a tanh-squashed action in [-1, 1]^3 to CarRacing's actual action bounds."""
    return ACTION_LOW + (tanh_action + 1.0) * 0.5 * (ACTION_HIGH - ACTION_LOW)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.out_features = 32 * 9 * 9

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.view(x.size(0), -1)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(GaussianPolicy, self).__init__()
        self.encoder = CNNEncoder(state_dim[0])
        self.fc1 = nn.Linear(self.encoder.out_features, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(self.fc1(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.encoder = CNNEncoder(state_dim[0])
        self.fc1 = nn.Linear(self.encoder.out_features + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = self.encoder(x)
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ==================== Replay Buffer ====================

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, action_dim), dtype=np.float32)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.ns = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def update(self, s, a, r, ns, done):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.ns[self.ptr] = ns
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def update_batch(self, s, a, r, ns, done):
        """Insert N transitions at once (from N parallel envs)."""
        for i in range(len(r)):
            self.update(s[i], a[i], r[i], ns[i], done[i])

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.s[ind]),
            torch.FloatTensor(self.a[ind]),
            torch.FloatTensor(self.r[ind]),
            torch.FloatTensor(self.ns[ind]),
            torch.FloatTensor(self.done[ind]),
        )


# ==================== SAC Agent ====================

class SAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        warmup_steps=1000,
        buffer_size=int(1e5),
        updates_per_batch=2,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        # How many update_model() gradient steps to run per process_batch() call
        # (i.e. per vec-step of N_ENVS parallel transitions). A 1:1 update-to-data
        # ratio (matching the single-env design) would be N_ENVS updates per call,
        # but that makes GPU updates dominate wall-clock time and mostly cancels
        # out the speedup from parallel env collection. Default here is a 1/4
        # ratio (e.g. 2 updates for N_ENVS=8): a middle ground between training
        # intensity and wall-clock speed.
        self.updates_per_batch = updates_per_batch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = GaussianPolicy(state_dim, action_dim).to(self.device)

        self.critic1 = QNetwork(state_dim, action_dim).to(self.device)
        self.critic2 = QNetwork(state_dim, action_dim).to(self.device)
        self.critic1_target = QNetwork(state_dim, action_dim).to(self.device)
        self.critic2_target = QNetwork(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in self.critic1_target.parameters():
            p.requires_grad_(False)
        for p in self.critic2_target.parameters():
            p.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr)

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        self.total_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, training=True):
        """Single-state action selection (used by evaluate() / visualize_sac.py)."""
        return self.select_action_batch(state[None], training=training)[0]

    def select_action_batch(self, states, training=True):
        """Batched action selection for N parallel envs. states: (N, C, H, W)."""
        if training and self.total_steps < self.warmup_steps:
            return np.random.uniform(-1.0, 1.0, size=(states.shape[0], self.action_dim)).astype(np.float32)

        states_t = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            action, _, mean_action = self.actor.sample(states_t)
        chosen = action if training else mean_action
        return chosen.cpu().numpy()

    def update_model(self):
        s, a, r, ns, done = map(lambda x: x.to(self.device), self.replay_buffer.sample(self.batch_size))

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(ns)
            target_q1 = self.critic1_target(ns, next_action)
            target_q2 = self.critic2_target(ns, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = r + (1 - done) * self.gamma * target_q

        critic1_loss = F.mse_loss(self.critic1(s, a), target_value)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        critic2_loss = F.mse_loss(self.critic2(s, a), target_value)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        new_action, log_prob, _ = self.actor.sample(s)
        q_new = torch.min(self.critic1(s, new_action), self.critic2(s, new_action))
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.mul_(1 - self.tau).add_(self.tau * param.data)

    def process_batch(self, transitions):
        """transitions: (s, a, r, ns, done), each of length N (from N parallel envs).

        Runs self.updates_per_batch update_model() calls per call (see comment on
        that attribute in __init__) rather than one call per collected transition.
        """
        s, a, r, ns, done = transitions
        n = len(r)
        self.total_steps += n
        self.replay_buffer.update_batch(s, a, r, ns, done)
        if self.replay_buffer.size >= self.batch_size and self.total_steps > self.warmup_steps:
            for _ in range(self.updates_per_batch):
                self.update_model()


def evaluate(agent, n_evals=3):
    eval_env = gym.make('CarRacing-v2', continuous=True, render_mode='rgb_array')
    eval_env = ImageEnv(eval_env)

    total_rewards = 0.0
    for _ in range(n_evals):
        s, _ = eval_env.reset()
        done = False
        truncated = False
        while not done and not truncated:
            a = agent.select_action(s, training=False)
            s, r, done, truncated, _ = eval_env.step(rescale_action(a))
            total_rewards += r
    eval_env.close()
    return total_rewards / n_evals


# ==================== Checkpoint save/load (for best/, last/, and resume) ====================

def save_checkpoint(agent, dir_path):
    """Save actor/critic/log_alpha weights + optimizer states + total_steps into dir_path."""
    os.makedirs(dir_path, exist_ok=True)
    torch.save(agent.actor.state_dict(), os.path.join(dir_path, 'sac_actor.pt'))
    torch.save(agent.critic1.state_dict(), os.path.join(dir_path, 'sac_critic1.pt'))
    torch.save(agent.critic2.state_dict(), os.path.join(dir_path, 'sac_critic2.pt'))
    torch.save(agent.log_alpha, os.path.join(dir_path, 'sac_log_alpha.pt'))
    # Optimizer momentum/variance state + step counter, so a resume can actually
    # continue training (not just warm-start the network weights from scratch
    # optimizer state, which tends to cause a transient loss/return spike).
    torch.save({
        'actor_optimizer': agent.actor_optimizer.state_dict(),
        'critic1_optimizer': agent.critic1_optimizer.state_dict(),
        'critic2_optimizer': agent.critic2_optimizer.state_dict(),
        'alpha_optimizer': agent.alpha_optimizer.state_dict(),
        'total_steps': agent.total_steps,
    }, os.path.join(dir_path, 'sac_train_state.pt'))


def load_checkpoint(agent, dir_path):
    """Load weights (+ optimizer state + total_steps, if present) from dir_path into `agent`, in place."""
    agent.actor.load_state_dict(torch.load(os.path.join(dir_path, 'sac_actor.pt'), map_location=agent.device))
    agent.critic1.load_state_dict(torch.load(os.path.join(dir_path, 'sac_critic1.pt'), map_location=agent.device))
    agent.critic2.load_state_dict(torch.load(os.path.join(dir_path, 'sac_critic2.pt'), map_location=agent.device))
    agent.critic1_target.load_state_dict(agent.critic1.state_dict())
    agent.critic2_target.load_state_dict(agent.critic2.state_dict())

    # Copy into the *existing* log_alpha tensor in place rather than replacing
    # the attribute outright -- agent.alpha_optimizer already holds a reference
    # to the original tensor object, and swapping it out would desync them.
    loaded_log_alpha = torch.load(os.path.join(dir_path, 'sac_log_alpha.pt'), map_location=agent.device)
    with torch.no_grad():
        agent.log_alpha.copy_(loaded_log_alpha)

    train_state_path = os.path.join(dir_path, 'sac_train_state.pt')
    if os.path.exists(train_state_path):
        state = torch.load(train_state_path, map_location=agent.device)
        agent.actor_optimizer.load_state_dict(state['actor_optimizer'])
        agent.critic1_optimizer.load_state_dict(state['critic1_optimizer'])
        agent.critic2_optimizer.load_state_dict(state['critic2_optimizer'])
        agent.alpha_optimizer.load_state_dict(state['alpha_optimizer'])
        agent.total_steps = state['total_steps']
    else:
        # Checkpoints saved by older versions of this script (or migrated by
        # hand) may only have the 4 weight files. Weights still load fine, but
        # optimizer momentum and total_steps can't be recovered -- this is a
        # warm start (fresh optimizer state, replay buffer, and step=0), not
        # an exact resume of the original run.
        print(f"WARNING: no sac_train_state.pt in '{dir_path}' -- optimizer state and "
              f"total_steps were NOT restored, only network weights. This is a warm "
              f"start, not an exact resume of that run.")


# ==================== Main (guarded: required for Windows multiprocessing) ====================

if __name__ == "__main__":
    N_ENVS = 8
    max_steps = int(2000000)
    eval_interval = 2000
    state_dim = (4, 84, 84)
    action_dim = 3  # [steer, gas, brake]

    # ---- Resume from a checkpoint instead of training from scratch ----
    # Set to 'last' to continue the most recent run, 'best' to continue from
    # the best-eval checkpoint seen so far, or any other checkpoint directory
    # (must contain sac_actor.pt / sac_critic1.pt / sac_critic2.pt / sac_log_alpha.pt,
    # and ideally sac_train_state.pt -- see load_checkpoint()'s docstring/warning).
    # Leave as None to train from scratch.
    #   RESUME_FROM = 'last'
    #   RESUME_FROM = 'best'
    RESUME_FROM = 'last'

    _t_start = time.time()

    def _progress_print(msg):
        print(f"[{time.time() - _t_start:8.1f}s] {msg}", flush=True)

    _progress_print(f"=== PARALLEL TRAINING START (max_steps={max_steps}, N_ENVS={N_ENVS}) ===")

    vec_env = AsyncVectorEnv([make_env for _ in range(N_ENVS)])
    # buffer_size lowered from the 1e5 default: with state_dim=(4,84,84) float32,
    # each of the buffer's `s`/`ns` arrays costs ~110KB per slot, so 1e5 slots
    # needs ~21GB combined -- too much for this machine (16GB RAM, 8 parallel
    # env worker processes also competing for it). 10000 slots costs ~2.2GB.
    agent = SAC(state_dim, action_dim, buffer_size=10000)
    _progress_print(f"agent device: {agent.device}")

    # NOTE: resuming only restores network weights + optimizer state + total_steps
    # (see save/load_checkpoint above). The replay buffer and history/plot are NOT
    # restored -- the buffer just refills as training continues, and
    # training_history.json/training_progress.png start a fresh curve from this
    # run (copy the old training_history.json aside first if you want to keep it).
    if RESUME_FROM is not None:
        # Don't hard-crash just because RESUME_FROM points at an empty/nonexistent
        # dir (e.g. 'best' before any best/ checkpoint has ever been saved) --
        # fall back to training from scratch instead, with a clear warning.
        if os.path.exists(os.path.join(RESUME_FROM, 'sac_actor.pt')):
            load_checkpoint(agent, RESUME_FROM)
            _progress_print(f"resumed from '{RESUME_FROM}' at total_steps={agent.total_steps}")
        else:
            _progress_print(f"WARNING: RESUME_FROM='{RESUME_FROM}' has no sac_actor.pt -- "
                             f"training from scratch instead.")

    # Best-eval checkpoint tracking. Reads any existing best/best_meta.json so a
    # resumed run doesn't immediately clobber a better checkpoint from a
    # previous run with a worse one.
    best_avg_return = float('-inf')
    best_meta_path = os.path.join('best', 'best_meta.json')
    if os.path.exists(best_meta_path):
        with open(best_meta_path) as f:
            best_avg_return = json.load(f)['avg_return']
        _progress_print(f"loaded existing best AvgReturn={best_avg_return:.3f} from {best_meta_path}")

    # Load the existing training_history.json (if any) so the curve/plot keeps
    # appending across a resume instead of restarting from Step 0 -- otherwise
    # training_progress.png would get overwritten with a fresh, truncated plot
    # the moment the next checkpoint fires.
    history = {'Step': [], 'AvgReturn': []}
    history_path = 'training_history.json'
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        _progress_print(f"loaded existing {history_path} ({len(history['Step'])} points, "
                         f"last Step={history['Step'][-1] if history['Step'] else 0})")

    # If we resumed from a checkpoint that had no sac_train_state.pt (e.g. a
    # weights-only/migrated checkpoint -- see load_checkpoint()'s warning),
    # agent.total_steps comes back as 0 even though the model isn't actually
    # fresh. Fall back to the loaded history's last Step so the step counter
    # (and thus the plot's x-axis, and next_progress/next_checkpoint below)
    # keeps counting up from where the run visually left off, instead of
    # restarting at 0.
    if RESUME_FROM is not None and agent.total_steps == 0 and history['Step']:
        agent.total_steps = history['Step'][-1]
        _progress_print(f"no total_steps in checkpoint -- resuming step counter "
                         f"from training_history.json instead: {agent.total_steps}")

    # Computed relative to agent.total_steps (0 for a fresh run, >0 if resumed)
    # so these don't fire immediately/skip ahead when resuming mid-training.
    next_progress = (agent.total_steps // 1000 + 1) * 1000
    next_checkpoint = (agent.total_steps // eval_interval + 1) * eval_interval

    try:
        obs, _ = vec_env.reset()
        while agent.total_steps < max_steps:
            actions = agent.select_action_batch(obs)
            env_actions = rescale_action(actions)
            next_obs, rewards, terminated, truncated, infos = vec_env.step(env_actions)

            agent.process_batch((
                obs,
                actions,
                np.asarray(rewards, dtype=np.float32),
                next_obs,
                np.asarray(terminated, dtype=np.float32),
            ))
            obs = next_obs

            if agent.total_steps >= next_progress:
                _progress_print(f"step {agent.total_steps}/{max_steps}, buffer size {agent.replay_buffer.size}")
                next_progress += 1000

            if agent.total_steps >= next_checkpoint:
                avg_return = evaluate(agent)
                history['Step'].append(agent.total_steps)
                history['AvgReturn'].append(avg_return)

                plt.figure(figsize=(8, 5))
                plt.plot(history['Step'], history['AvgReturn'], 'r-')
                plt.xlabel('Step', fontsize=16)
                plt.ylabel('AvgReturn', fontsize=16)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.grid(axis='y')
                plt.savefig('training_progress.png')
                plt.close()

                # 'last/' always holds the most recent checkpoint (used by RESUME_FROM='last').
                save_checkpoint(agent, 'last')

                # 'best/' only gets overwritten when eval actually improves, so it never
                # regresses even if a later checkpoint (like this one) is worse.
                if avg_return > best_avg_return:
                    best_avg_return = avg_return
                    save_checkpoint(agent, 'best')
                    with open(best_meta_path, 'w') as f:
                        json.dump({'step': agent.total_steps, 'avg_return': best_avg_return}, f, indent=2)
                    _progress_print(f"[best] step {agent.total_steps} AvgReturn {avg_return:.3f} -> saved to best/")

                with open('training_history.json', 'w') as f:
                    json.dump(history, f, indent=2)

                _progress_print(f"[checkpoint] step {agent.total_steps} AvgReturn {avg_return:.3f}")
                next_checkpoint += eval_interval
    finally:
        vec_env.close()

    _progress_print("TRAINING LOOP DONE")
    print("final history:", history, flush=True)
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    save_checkpoint(agent, 'last')

    _progress_print("ALL DONE")
