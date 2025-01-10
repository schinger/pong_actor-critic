# 从头理解策略梯度（Policy Gradient）算法及定理

## 1. 算法介绍
强化学习有两大类方法，一类是基于值函数的方法，另一类是基于策略的方法。策略梯度（Policy Gradient）算法属于基于策略的方法，它将策略参数化，基于此参数定义一个目标函数$J(\theta)$，通过梯度上升的方式更新策略参数，使得目标函数最大化。

$$
\theta_{t+1} = \theta_t + \alpha \nabla_{\theta} J(\theta_t)
$$
一般地，我们将$J(\theta)$定义为在策略$\pi_{\theta}$下初始状态的价值函数：
$$
J(\theta) \doteq V^{\pi_\theta}(s_0)
$$
幸运的是，下一节的策略梯度定理告诉我们，$J(\theta)$的梯度有一个简洁的形式：
$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \mathbb{E}_\pi [Q^\pi(s, a) \nabla_\theta \log \pi_\theta(a \vert s)] & \\
&= \mathbb{E}_\pi [G_t \nabla_\theta \log \pi_\theta(A_t \vert S_t)] & \scriptstyle{\text{; Because } Q^\pi(S_t, A_t) = \mathbb{E}_\pi[G_t \vert S_t, A_t]}
\end{aligned}
$$
其中，$Q^\pi(s, a)$是在策略$\pi$下状态-动作对$(s, a)$的价值函数，$G_t$是从时刻$t$开始的回报。据此，我们可以通过模特卡罗对S, A, R序列进行采样，计算梯度，更新策略参数。这就是Reinforce算法（考虑奖励折算系数$\gamma$的情况。下文中为了论述的简洁，我们假设$\gamma=1$）：


- Input： 随机初始化的策略$\pi_{\theta}(a|s)$，学习率$\alpha$

- 执行循环：
    - 依据策略$\pi_{\theta}$采样S, A, R序列：$S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T$
    - 按时间步t=0, 1, ..., T-1循环执行：
        - $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$
        - $\theta = \theta + \alpha \gamma^t G_t \nabla_{\theta} \log \pi_{\theta}(A_t|S_t)$

Reinforce算法有非常直观的理解：如果某个动作的回报较高，那么相应的增大这个动作的概率（朝其梯度方向更新）；反之如果回报为负，则减小这个动作的概率（朝其梯度反方向更新）。下一节我们证明算法背后的策略梯度定理。
## 2. 定理证明
我们直接对值函数的梯度进行展开：
$$
\begin{aligned}
& \nabla_\theta V^\pi(s) \\
=& \nabla_\theta \Big(\sum_{a \in \mathcal{A}} \pi_\theta(a \vert s)Q^\pi(s, a) \Big) & \\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) {\nabla_\theta Q^\pi(s, a)} \Big) & \scriptstyle{\text{; Derivative product rule.}} \\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) {\nabla_\theta \sum_{s', r} P(s',r \vert s,a)(r + V^\pi(s'))} \Big) & \scriptstyle{\text{; Extend } Q^\pi \text{ with future state value.}} \\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) {\sum_{s', r} P(s',r \vert s,a) \nabla_\theta V^\pi(s')} \Big) & \scriptstyle{P(s',r \vert s,a) \text{ or } r \text{ is not a func of }\theta}\\
=& \sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) + \pi_\theta(a \vert s) {\sum_{s'} P(s' \vert s,a) \nabla_\theta V^\pi(s')} \Big) & \scriptstyle{\text{; Because }  P(s' \vert s, a) = \sum_r P(s', r \vert s, a)}
\end{aligned}
$$
可以看出，$\nabla_\theta V^\pi(s)$的计算涉及到下个状态的值函数梯度$\nabla_\theta V^\pi(s')$，这是一个递归的过程。为了方便后续推导，我们先暂停一下引入一些记号：
我们令$\phi(s) = \sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a)$; 令$\rho^\pi(s \to x, k)$表示在策略$\pi$下从状态$s$开始，经过$k$步转移到状态$x$的概率，我们立即得到如下关系：

-  $\rho^\pi(s \to s, k=0) = 1$
- $\rho^\pi(s \to s', k=1) = \sum_a \pi_\theta(a \vert s) P(s' \vert s, a)$
- $\rho^\pi(s \to x, k+1) = \sum_{s'} \rho^\pi(s \to s', k) \rho^\pi(s' \to x, 1)$

现在，让我们继续$\nabla_\theta V^\pi(s)$的推导：
$$
\begin{aligned}
& {\nabla_\theta V^\pi(s)} \\
=& \phi(s) + \sum_a \pi_\theta(a \vert s) \sum_{s'} P(s' \vert s,a) {\nabla_\theta V^\pi(s')} \\
=& \phi(s) + \sum_{s'} \sum_a \pi_\theta(a \vert s) P(s' \vert s,a) {\nabla_\theta V^\pi(s')} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) {\nabla_\theta V^\pi(s')} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) {\sum_{a \in \mathcal{A}} \Big( \nabla_\theta \pi_\theta(a \vert s')Q^\pi(s', a) + \pi_\theta(a \vert s') \sum_{s'} P(s'' \vert s',a) \nabla_\theta V^\pi(s'') \Big)} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) {[ \phi(s') + \sum_{s''} \rho^\pi(s' \to s'', 1) \nabla_\theta V^\pi(s'')]} \\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \phi(s') + \sum_{s''} \rho^\pi(s \to s'', 2){\nabla_\theta V^\pi(s'')} \scriptstyle{\text{ ; Consider }s'\text{ as the middle point for }s \to s''}\\
=& \phi(s) + \sum_{s'} \rho^\pi(s \to s', 1) \phi(s') + \sum_{s''} \rho^\pi(s \to s'', 2)\phi(s'') + \sum_{s'''} \rho^\pi(s \to s''', 3){\nabla_\theta V^\pi(s''')} \\
=& \dots \scriptstyle{\text{; Repeatedly unrolling the part of }\nabla_\theta V^\pi(.)} \\
=& \sum_{x\in\mathcal{S}}\sum_{k=0}^\infty \rho^\pi(s \to x, k) \phi(x)
\end{aligned}
$$
将这一结论应用于$J(\theta)$的梯度：
$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \nabla_\theta V^\pi(s_0) & \scriptstyle{\text{; Starting from a random state } s_0} \\
&= \sum_{s}{\sum_{k=0}^\infty \rho^\pi(s_0 \to s, k)} \phi(s) &\scriptstyle{\text{; Let }{\eta(s) = \sum_{k=0}^\infty \rho^\pi(s_0 \to s, k)}} \\
&= \sum_{s}\eta(s) \phi(s) & \\
&= \Big( {\sum_s \eta(s)} \Big)\sum_{s}\frac{\eta(s)}{\sum_s \eta(s)} \phi(s) & \scriptstyle{\text{; Normalize } \eta(s), s\in\mathcal{S} \text{ to be a probability distribution.}}\\
&\propto \sum_s \frac{\eta(s)}{\sum_s \eta(s)} \phi(s) & \scriptstyle{\sum_s \eta(s)\text{  is a constant}} \\
&= \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) & \scriptstyle{d^\pi(s) = \frac{\eta(s)}{\sum_s \eta(s)}\text{ is on-policy state distribution.}} \\
&= \sum_{s } d^\pi(s) \sum_{a } \pi_\theta(a \vert s) Q^\pi(s, a) \frac{\nabla_\theta \pi_\theta(a \vert s)}{\pi_\theta(a \vert s)} &\\
&= \mathbb{E}_\pi [Q^\pi(s, a) \nabla_\theta \log \pi_\theta(a \vert s)] 
\end{aligned}
$$
证毕。
## 3. 广泛形式
朴素的策略梯度是unbiased的，但是方差较大，很多工作通过引入状态s的函数作为baseline：$b(s)$来减小方差，同时保持unbiased特性。由于：
$$
\sum_a b(s) \nabla_\theta \pi_\theta(a \vert s) = b(s) \nabla_\theta\sum_a  \pi_\theta(a \vert s) = b(s) \nabla_\theta 1 = 0
$$
所以可以将baseline加入到策略梯度中：
$$
\begin{aligned}
\nabla_\theta J(\theta)
&\propto \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \vert s)Q^\pi(s, a) \\
&= \sum_s d^\pi(s) \sum_a \nabla_\theta \pi_\theta(a \vert s)(Q^\pi(s, a) - b(s)) \\
&= \sum_s d^\pi(s) \sum_a \pi_\theta(a \vert s)(Q^\pi(s, a) - b(s)) \frac{\nabla_\theta \pi_\theta(a \vert s)}{\pi_\theta(a \vert s)} \\
&= \mathbb{E}_\pi [(Q^\pi(s, a) - b(s)) \nabla_\theta \log \pi_\theta(a \vert s)] 
\end{aligned}
$$
一个直观的做法是将baseline设置为状态价值函数$b(s)=V^\pi(s)$，其物理意义是：如果某个动作的价值高于平均水平，那么相应的增大这个动作的概率；反之如果价值低于平均水平，则减小这个动作的概率。

更进一步，我们可以将Policy Gradient写成更广泛的形式：
$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi [\Phi^\pi(s,a) \nabla_\theta \log \pi_\theta(a \vert s)]
$$
其中，$\Phi^\pi(s,a)$可以是如下形式：
- $G$ following $(s,a)$
- $G - b(s)$
- $Q^\pi(s,a)$
- $Q^\pi(s,a) - V^\pi(s)$
- Advantage： $A^\pi(s,a)$
- TD residual: $r + \gamma V^\pi(s') - V^\pi(s)$

我们上面的论述都是基于离散动作空间和随机策略的情形，对于连续动作空间以及确定性策略，我们也可以推导出类似的策略梯度定理。再次引入三个记号：
- $\rho_0(s)$: 初始状态分布
- $\rho^\pi(s \to s', k)$: 在策略$\pi$下从状态$s$开始，经过$k$步转移到状态$s'$的概率密度
- $\rho^\pi(s') = \int_\mathcal{S} \sum_{k=1}^\infty \gamma^{k-1} \rho_0(s) \rho^\pi(s \to s', k) ds$： 在策略$\pi$下状态$s'$的折算分布

则随机策略$\pi_\theta(a|s)$连续空间的策略梯度定理为：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int_S \rho^\pi(s) \int_A \nabla_\theta \pi_\theta(a|s) Q^\pi(s, a) dads \\
&= \mathbb{E}_{s \sim \rho^\pi, a \sim \pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) Q^\pi(s, a)]
\end{aligned}
$$
确定性策略$a=\mu_\theta(s)$连续空间的策略梯度定理为：
$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int_S \rho^\mu(s) \nabla_\theta Q^\mu(s, \mu_\theta(s)) ds \\
 &= \int_S \rho^\mu(s) \nabla_\theta \mu_\theta(s) 
\nabla_a Q^\mu(s, a) \vert_{a=\mu_\theta(s)} ds \\
&= \mathbb{E}_{s \sim \rho^\mu} [\nabla_\theta \mu_\theta(s)\nabla_a Q^\mu(s, a) \vert_{a=\mu_\theta(s)}]

\end{aligned}
$$
即让参数$\theta$朝着使得$Q$函数增大的方向更新。
## 4. 代码实例

我们以Pong（乒乓球游戏）为例，[用一百多行代码基于numpy实现actor-critic策略梯度算法](https://github.com/schinger/pong_actor-critic)。actor和critic均为简单的feedforward网络，手工实现前向传播和反向传播。用Expotential Moving Average (EMA) 更新Target Value Network: $V^{tar}(s)$。采用k-step TD error作为Advantage，并逐步缩小k值：
$$
A_t = \sum_{i=0}^{k-1} \gamma^i r_{t+i} + \gamma^k V^{tar}(s_{t+k}) - V(s_t)
$$
策略梯度为：
$$
\nabla_\theta J(\theta) = \mathbb{E}_\pi [A_t \nabla_\theta \log \pi_\theta(a_t \vert s_t)]
$$
Github地址：https://github.com/schinger/pong_actor-critic
