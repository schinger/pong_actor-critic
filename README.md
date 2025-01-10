# pong_actor-critic
Train an agent with (Stochastic) Policy Gradients (Actor-Critic) on Pong. Use OpenAI Gym.

(Run success in Aug 2023. gym 0.26.2, python 3.7.4, windows 11)

- background intro. http://karpathy.github.io/2016/05/31/rl/
- the original REINFORCE counterpart https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
- within 200,000 round can beat the build-in AI player. when training finished, you can turn on `resume` and `render` to see the playing animation
- one hundred lines (~160) python, only need numpy, manually implement the forward and backward pass
- Actor and Critic both use simple feedforward neural network with one hidden layer
- k-step TD error as the advantage, the policy gradient is modulated by the advantage (line 130)
- update target Value Network by moving average (line 149)