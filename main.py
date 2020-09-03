from acme import environment_loop
from acme import networks
from acme.adders import reverb as adders
from acme.agents import actors_tf2 as actors
from acme.datasets import reverb as datasets
from acme.wrappers import gym_wrapper
from acme import specs
from acme import wrappers
from acme.agents import d4pg
from acme.utils import tf2_utils
from acme.utils import loggers

import gym
import dm_env
import matplotlib.pyplot as plt
import numpy as np 
import reverb
import sonnet as snt
import tensorflow as tf

from IPython.display import clear_output
clear_output()

# Load Environment:

env = gym_wrapper.GymWrapper(gym.make('MountainCarContinuous-v0'))
env = wrappers.SinglePrecisionWrapper(env)

env.environment.render(mode='rgb_array')

def render(env):
    return env.environment.render(mode='rgb_array')

environment_spec = specs.make_environment_spec(env)

# Create D4PG Agent:

# Get total number of action dimensions from action spec

num_dimensions = np.prod(environment_spec.actions.shape, dtype=int)

# Create shared observation network:
observation_network = tf2_utils.batch_concat

policy_network = snt.Sequential([
	networks.LayerNormMLP((256, 256, 256), activate_final=True),
	networks.NearZeroInitializedLinear(num_dimensions),
	networks.TanhToSpec(environment_spec.actions)
	])

# Create the distributional critic network:
critic_network = snt.Sequential([
	networks.CriticMultiplexer(),
	networks.LayerNormMLP((512, 512, 256), activate_final=True),
	networks.DiscreteValuedHead(vmin=-150., vmax=150., num_atoms=51)])

# Create logger for agent diagnostics:
agent_logger = loggers.TerminalLogger(label='agent', time_delta=10)

# Create D4PG Agent:
agent = d4pg.D4PG(
	environment_spec=environment_spec,
	policy_network=policy_network,
	critic_network=critic_network,
	observation_network=observation_network,
	logger=agent_logger,
	checkpoint=False
)

# Run Training Loop

env_loop_logger = loggers.TerminalLogger(label='env_loop', time_delta=10)

env_loop = environment_loop.EnvironmentLoop(env, agent, logger=env_loop_logger)
env_loop.run(num_episodes=5)


# Displaying:

import pyvirtualdisplay
import imageio
import base64
import copy
import IPython

display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

clear_output()

def display_video(frames, filename='temp.mp4'):
	# Write video:
	with imageio.get_writer(filename, fps=60) as video:
		for frame in frames:
			video.append_data(frame)

	# Read Video:
	video = open(filename, 'rb').read()
	b64_video = base64.b64encode(video)
	video_tag = ('<video width="320" height="240" controls alt="test"'
				'src="data:video/mp4;base64, {0}">').format(b64_video.decode())
	
	return IPython.display.HTML(video_tag)

# Run the actor in the environment for desired number of steps:
frames = []
num_steps = 100
timestep = env.reset()

for _ in range(num_steps):
	frames.append(render(env))
	action = agent.select_action(timestep.observation)
	timestep = env.step(action)

# Save video:
display_video(np.array(frames))