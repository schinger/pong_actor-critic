# pong_actor-critic
Trains an agent with (stochastic) Policy Gradients(actor-critic) on Pong. Uses OpenAI Gym.

(Run success in Aug 2023. gym 0.26.2, python 3.7.4, windows 11)

- background intro. http://karpathy.github.io/2016/05/31/rl/
- the original REINFORCE counterpart https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
- refer to Sutton's book and Silver's paper, slides
- one hundred lines(160) python, only need numpy
- very slow!(within 200,000 round can beat the build-in AI player)
