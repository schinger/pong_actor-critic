""" Trains an agent with (stochastic) Policy Gradients(actor-critic) on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle as pickle
import gym

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 200 #
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
  model = pickle.load(open('save.ac', 'rb'))
else:
  model = {}
  model['W1_policy'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2_policy'] = np.random.randn(H) / np.sqrt(H)
  model['W1_value'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2_value'] = np.random.randn(H) / np.sqrt(H)
  
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def forward(x,modelType):
  h = np.dot(model['W1_'+modelType], x)
  h[h<0] = 0 # ReLU nonlinearity
  out = np.dot(model['W2_'+modelType], h)
  if modelType == 'policy':
    out = sigmoid(out)
  return out, h

def backward(eph,epx,epd,modelType):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(eph.T, epd).ravel()
  dh = np.outer(epd, model['W2_'+modelType])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1_'+modelType:dW1, 'W2_'+modelType:dW2}

env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None # used in computing the difference frame
v_next,h_v_next=None,None
xs,h_ps,h_vs,dlogps,dv = [],[],[],[],[]
running_reward = None
reward_sum = 0
round_number = 0
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  x = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  aprob, h_p = forward(x,'policy')
  action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

  if v_next is None:
    v,h_v = forward(x,'value') 
  else:
    v,h_v = v_next,h_v_next
  # record various intermediates (needed later for backprop)
  xs.append(x) # observation
  h_ps.append(h_p) # hidden state
  h_vs.append(h_v)
  y = 1 if action == 2 else 0 # a "fake label"
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, info = env.step(action)
  reward_sum += reward

  v_next,h_v_next=forward(prepro(observation)-prev_x,'value')
  dv.append(reward+gamma*v_next-v) 

  if reward != 0: 
    round_number += 1

    # stack together all inputs, hidden states, action gradients, and td for this episode
    epx = np.vstack(xs)
    eph_p = np.vstack(h_ps)
    eph_v = np.vstack(h_vs)
    epdlogp = np.vstack(dlogps)
    epv = np.vstack(dv)
    xs,h_ps,h_vs,dlogps,dv = [],[],[],[],[] # reset array memory


    discounted_epv = epv * np.vstack([gamma**i for i in range(len(epv))])
    epdlogp *= discounted_epv # modulate the gradient with advantage (PG magic happens right here.)
    grad_p = backward(eph_p,epx,epdlogp,'policy')
    grad_v = backward(eph_v,epx,epv,'value')
    grad = dict(grad_p,**grad_v)

    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size
    if round_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    print (('round %d game finished, reward: %f' % (round_number, reward)) + ('' if reward == -1 else ' !!!!!!!!'))

  # boring book-keeping
  if done:
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    reward_sum = 0
    observation = env.reset() # reset env
    prev_x = None
    v_next,h_v_next=None,None
  if round_number % 2000 == 0: pickle.dump(model, open('save.ac', 'wb'))