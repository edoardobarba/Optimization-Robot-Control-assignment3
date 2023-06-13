import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform
from collections import deque
from pendulum import *
import wandb
import os
import matplotlib.pyplot as plt
import ex_0_policy_evaluation_sol_prof
from dpendulum import DPendulum


from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class DQN:
    def __init__(self, nx, nu):
        self.nx = nx
        self.nu = nu
        self.max_size = 30000
        self.memory_counter = 0
        self.scrolling_index = 0
        self.replay_memory = np.zeros(self.max_size, dtype=object)
        self.batch_size = 64
        self.DISCOUNT = 0.99 #gamma
        self.initial_exploration = 0.95
        self.final_exploration = 0.001
        self.exploration_decay = 0.002
        self.QVALUE_LEARNING_RATE = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)

        self.update_counter = 0

        self.umin = -2
        self.umax = 2
        self.size_action_space = 17
        self.actions = np.linspace(self.umin, self.umax, self.size_action_space)
        print("Possible actions: ", self.actions)

        self.Q = self.get_critic(self.nx, self.nu)
        self.target_Q = self.get_critic(self.nx, self.nu)


    def get_critic(self, nx, nu):
        ''' Create the neural network to represent the Q function '''
        inputs = layers.Input(shape=(nx))
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        state_out4 = layers.Dense(64, activation="relu")(state_out3)
        outputs = layers.Dense(self.size_action_space)(state_out4) 

        model = tf.keras.Model(inputs, outputs)

        return model

    def save_experience(self, x, u, c, next_x, is_last):
        index = self.scrolling_index % self.max_size
        self.replay_memory[index] = (x, u, c, next_x, is_last)
        self.scrolling_index += 1
        if self.memory_counter != self.max_size:
            self.memory_counter += 1


    # def update(self, xu_batch, cost_batch, xu_next_batch):
    #     batch = np.random.choice(self.replay_memory, self.batch_size, replace=False)
    #     states, actions, costs, next_states, is_lasts = zip(*[self.memory[i] for i in batch])
    #     ''' Update the weights of the Q network using the specified batch of data '''
    #     # all inputs are tf tensors
    #     with tf.GradientTape() as tape:         
    #         # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
    #         # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
    #         # Tensors can be manually watched by invoking the watch method on this context manager.
    #         target_values = Q_target(xu_next_batch, training=True)   
    #         # Compute 1-step targets for the critic loss
    #         y = cost_batch + DISCOUNT*target_values                            
    #         # Compute batch of Values associated to the sampled batch of states
    #         Q_value = Q(xu_batch, training=True)                         
    #         # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
    #         Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))  
    #     # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    #     Q_grad = tape.gradient(Q_loss, Q.trainable_variables)          
    #     # Update the critic backpropagating the gradients
    #     critic_optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))

    def get_epsilon_action(self, x, episode, exp = True):
        final_epsilon = 0.001
        initial_epsilon = 0.9
        decay_rate = 0.002
        
        if exp:
            #exponential
            epsilon = self.final_exploration + (self.initial_exploration-self.final_exploration) * np.exp(-self.exploration_decay*episode)
        else:
            #linear
            epsilon = max(self.final_exploration,self.initial_exploration - self.exploration_decay * episode)

        if(np.random.rand(1)<=epsilon):
            idx_u = np.random.choice(self.size_action_space)
        else:
            input = np.expand_dims(np.array(x), axis=0)
            q_values = self.Q(input)
            #print(q_values)
            idx_u = np.argmax(q_values)
            #print(idx_u)

        return idx_u
    
    def fill_replay_memory(self, env):
        for i in range(3000):
            index = self.scrolling_index % self.max_size
            x = env.reset()
            # print(x) 
            if self.nx == 2:   
                x = [x[0][0], x[1][0]]
            else:
                x = [x[0][0], x[0][1], x[1][0], x[1][1]]

            # else: 
            #     x = [x[0][0], x[0][1], x[1][0], x[1][1]]

            # print(x) 

            #print(x)
            idx_u = np.random.choice(self.size_action_space)
            #print("idx_u: ", idx_u)
            u = self.actions[idx_u]
            #print("u: ", u)
            next_x, cost = env.step(u)

            
            if self.nx == 2:   
                next_x = [next_x[0][0], next_x[1][0]]

            else: 
                next_x = [next_x[0][0], next_x[0][1], next_x[1][0], next_x[1][1]]

            is_last = False
            t = (x, idx_u, cost, next_x, is_last)
            self.replay_memory[index] = t
            self.scrolling_index += 1
            if self.memory_counter != self.max_size:
                self.memory_counter += 1




def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()





def train_dqn(dqn, env, episodes, critic_optimizer, save_dir, goon=False):
    if goon:
        Q = tf.keras.models.load_model(save_dir)
        Q_target = dqn.target_Q
    else:
        Q = dqn.Q
        Q_target = dqn.target_Q

    Q.summary()

    # Set initial weights of targets equal to those of the critic
    Q_target.set_weights(Q.get_weights())

    # # Set optimizer specifying the learning rates
    # critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)

    w = Q.get_weights()

    dqn.fill_replay_memory(env)
    #print(dqn.replay_memory[:54])

    total_reward_hist = []
    avg_reward_hist = []

    # new_configuration = dict(
    #         epochs = epochs,
    #         batch_size = dqn.batch_size,
    #         total_reward_hist = total_reward_hist,
    #         #avg_reward_hist = avg_reward_hist,
    #         dataset_name="adaptiope_reduced"
    #     )

    # Run your code.
    #wandb.init(project="DQN_pendulum", entity="bru0x11") #, configuration=new_configuration, save_code=True)
    wandb.login(key='3ec27bc14641f193a2a62ed18f343b1fba35484a')
    wandb.init(project="pendulum_DQN")
    for episode in range(1, episodes+1):
        #env.reset(np.array([np.pi,0.,0.,0.]))
        if env.nx == 4: 
            #env.reset(np.array([[np.pi, 0.],[0.,0.]]))
            env.reset()
            x = env.x
            x = np.array([[x[0][0], 0],[0,0]])
            #print(env.x)
        else:
            env.reset()
            #env.reset(np.array([[np.pi],[0.]]))
        n_steps = 400
        sum_costs = 0 
        for t in range(1,n_steps+1):
            x = env.x
            # print(x)
            if env.nx == 2:
                x = [x[0][0], x[1][0]]
            else:
                x = [x[0][0], x[0][1], x[1][0], x[1][1]]

            # print(x)

            idx_u = dqn.get_epsilon_action(x, episode)
            u = dqn.actions[idx_u]

            next_x, cost = env.step(u)
            if env.nx == 2:
                next_x = [next_x[0][0], next_x[1][0]]

            else:
                next_x = [next_x[0][0], next_x[0][1], next_x[1][0], next_x[1][1]]
            is_last = False
            if t==n_steps:
                is_last = True

            dqn.save_experience(x, idx_u, cost, next_x, is_last)

            indexes_batch = np.random.choice(dqn.memory_counter-1, dqn.batch_size, replace=False).tolist()

            current_batch = dqn.replay_memory[indexes_batch]
            
            states = []
            actions = []
            costs = []
            next_states = []
            is_lasts = []

            for t in current_batch:
                states.append(t[0])
                actions.append(t[1])
                costs.append(t[2])
                next_states.append(t[3])
                is_lasts.append(t[4])

            indices = np.arange(dqn.batch_size)

            with tf.GradientTape() as tape:         
                #print(np.shape(np.array(states)))
                #print("np.array(states)")
                #print(np.array(states))
                q_pred = Q(np.array(states), training = True)[indices, actions]
                #print(np.shape(np.array(next_states)))
                q_target_pred = Q_target(np.array(next_states))
                #print(np.shape(q_target_pred))

                q_target_pred = np.array(np.max(q_target_pred, axis=1))

                q_target_pred[is_lasts] = 0.0

                total_cost = costs + (dqn.DISCOUNT * q_target_pred)   
                # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
                Q_loss = tf.math.reduce_mean(tf.math.square(total_cost - q_pred))  

            # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
            Q_grad = tape.gradient(Q_loss, Q.trainable_variables)          
            # Update the critic backpropagating the gradients
            critic_optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))

            sum_costs += cost
            
            if(is_last):
                break
            
            if((dqn.update_counter % 2500) == 0):
                Q_target.set_weights(Q.get_weights())

            dqn.update_counter += 1

        total_reward_hist.append(sum_costs)
        avg_reward = np.average(total_reward_hist[-100:])
        avg_reward_hist.append(avg_reward)
        wandb.log({'episode': episode, 'total_reward': sum_costs, 'avg_reward': avg_reward})

        if episode%100 == 0:
            temp_save_dir = save_dir + f"{episode}_episode"
            print("Saving model...")
            Q.save(temp_save_dir)
            print(temp_save_dir)

        print("Episode :", episode, "sum_costs : {:.2f}".format(sum_costs), "Avg Reward : {:.2f}".format(avg_reward))
 

    Q.save(save_dir)
    wandb.finish()
    fig, ax = plt.subplots()
    t = np.arange(episodes)
    ax.plot(t, total_reward_hist, label="Total Reward")
    ax.plot(t, avg_reward_hist, label="Average Reward")
    ax.set_title("Reward vs Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    plt.show()



def test_dqn(env, episodes, save_dir):
    # wandb.login(key='3ec27bc14641f193a2a62ed18f343b1fba35484a')
    # wandb.init(project="pendulum_DQN")
    Q = tf.keras.models.load_model(save_dir)
    # Run your code.
    #wandb.init(project="DQN_pendulum", entity="bru0x11") #, configuration=new_configuration, save_code=True)
    total_reward_hist = []
    avg_reward_hist = []

    for episode in range(1, episodes+1):
        print("Episode: ", episode)
        if env.nx == 4: 
            env.reset(np.array([[np.pi, 0.],[0.,0.]]))
        else:
            #env.reset(np.array([[np.pi],[0.]]))
            env.reset()
        n_steps = 100
        tot_reward = 0 
        angles = []
        velocities = []
        torque = []
        for t in range(1,n_steps+1):
            x = env.x
            
            if env.nx == 2:
                x = [x[0][0], x[1][0]]
            else:
                x = [x[0][0], x[0][1], x[1][0], x[1][1]]

            angles.append(x[0])
            velocities.append(x[1])

            #print(x)
            input = np.expand_dims(np.array(x), axis=0)
            #print(input)
            #input = x.reshape((2,1))
            #print(input)
            q_values = Q(input)
            idx_u = np.argmax(q_values)
            u = dqn.actions[idx_u]
            torque.append(u)
            #print("u: ", u)
            next_x, reward = env.step(u)
            tot_reward += reward
            if(t==n_steps):
                last_reward = reward
            
            #if episode == 10:
                #wandb.log({'time_step': t, 'u': u, 'q':x[0], 'v':x[1]})

        #wandb.log({'episode': episode, 'reward': total_reward})
        
        #plot 

        timesteps = np.arange(len(angles))

        plt.rcParams["figure.figsize"] = [10, 7]
        plt.rcParams["figure.autolayout"] = True

        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.set_title("State graph")
        ax1.plot(timesteps, angles, color="red", label="angles")
        ax1.plot(timesteps, velocities, color="blue", label="velocities")
        ax1.legend()

        ax2.set_title("Torque graph")
        ax2.plot(timesteps, torque, color="blue", label="torque")
        ax2.legend()

        plt.show()


        total_reward_hist.append(tot_reward)
        avg_reward = np.average(total_reward_hist[:])
        avg_reward_hist.append(avg_reward)
        print("Episode :", episode, "total_reward : {:.2f}".format(tot_reward), "Avg Reward : {:.2f}".format(avg_reward))
    #wandb.finish()


def pi(env, x):
    save_dir = "models/single_pendulum/prof_reward/1000_episodes1000_episode_random"
    Q = tf.keras.models.load_model(save_dir)

    x = env.x
            
    # if env.nx == 2:
    #     x = [x[0][0], x[1][0]]
    # else:
    #     x = [x[0][0], x[0][1], x[1][0], x[1][1]]

    input = np.expand_dims(np.array(x), axis=0)

    q_values = Q(input)
    idx_u = np.argmax(q_values)
    u = dqn.actions[idx_u]

    return u




# Set the WANDB__EXECUTABLE environment variable to the path to a valid Python interpreter.
os.environ["WANDB__EXECUTABLE"] = "/usr/bin/python3"


save_dir = "models/"
njoints = 1
episodes = 1 #10/min
QVALUE_LEARNING_RATE = 1e-3
DISCOUNT = 0.99


if njoints == 1:
    save_dir += "single_pendulum/"
else:
    save_dir += "double_pendulum/"
save_dir += f"{episodes}_episodes"

env = Pendulum(njoints)
# Create critic and target NNs
if njoints == 1:
    nx = 2
    nu = 1
else:
    nx = 4
    nu = 2

dqn = DQN(nx, nu)

critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)

#save_dir = "models/double_pendulum/3000_episodes250_episode300_episode"
#train_dqn(dqn, env, episodes, critic_optimizer, save_dir, goon=False)

save_dir = "models/single_pendulum/prof_reward/1000_episodes1000_episode_random"
test_dqn(env, episodes, save_dir)





### --- Hyper paramaters
MAX_ITERS         = 200         # Max number of iterations
CONVERGENCE_THR   = 1e-4        # convergence threshold
NPRINT            = 5           # Print info every NPRINT iterations
PLOT              = True        # Plot the V table
DISCOUNT          = 0.99         # Discount factor 

### --- Environment
nq=21   # number of discretization steps for the joint angle q
nv=11   # number of discretization steps for the joint velocity v
nu=11   # number of discretization steps for the joint torque u
env = DPendulum(nq, nv, nu) # create the environment
V  = np.zeros([env.nx])     # V-table initialized to 0

# display policy behavior
#    render_policy(env, policy)

V = ex_0_policy_evaluation_sol_prof.policy_eval(env, DISCOUNT, pi, V, MAX_ITERS, CONVERGENCE_THR, PLOT, NPRINT)
    
