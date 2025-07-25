import gymnasium as gym
import dqn
import matplotlib.pyplot as plt
import torch

env = gym.make('CartPole-v1')

dqn = dqn.DQN(env.observation_space.shape[0], env.action_space.n)
dqn.train(env)

plt.plot(dqn.returns)
plt.title("Returns (episode length) over time")
plt.xlabel("Episode")
plt.ylabel("Duration")
plt.savefig("returns-2.png")

env = gym.make('CartPole-v1', render_mode="human")
dqn.online_net.eval()
with torch.inference_mode():
    dqn.render_and_run(env)