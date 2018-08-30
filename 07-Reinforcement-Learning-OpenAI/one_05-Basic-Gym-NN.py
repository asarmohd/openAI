import tensorflow as tf
import gym
import numpy as np
###############################################
######## PART ONE: NETWORK VARIABLES #########
#############################################

# Observation Space has 4 inputs
num_inputs = 2

num_hidden = 2

# Outputs the probability it should go left
num_outputs = 1

initializer = tf.contrib.layers.variance_scaling_initializer()


###############################################
######## PART TWO: NETWORK LAYERS #########
#############################################

X = tf.placeholder(tf.float32, shape=[None,num_inputs])
hidden_layer_one = tf.layers.dense(X,num_hidden,activation=tf.nn.relu,kernel_initializer=initializer)
hidden_layer_two = tf.layers.dense(hidden_layer_one,num_hidden,activation=tf.nn.relu,kernel_initializer=initializer)

# Probability to go left
output_layer = tf.layers.dense(hidden_layer_two,num_outputs,activation=tf.nn.tanh,kernel_initializer=initializer)

# [ Prob to go left , Prob to go right]
probabilties = tf.concat(axis=1, values=[output_layer])

# Sample 1 randomly based on probabilities
action = probabilties


init = tf.global_variables_initializer()



###############################################
######## PART THREE: SESSION #########
#############################################

saver = tf.train.Saver()

epi = 200
step_limit = 500
avg_steps = []
env = gym.make("MountainCarContinuous-v0")
with tf.Session() as sess:
    init.run()
    for i_episode in range(epi):
        obs = env.reset()
        totalreward = tf.Variable(0, dtype=tf.float32)
        y = tf.Variable(100, dtype=tf.float32)
        for step in range(step_limit):
            # env.render()
            action_val = action.eval(feed_dict={X: obs.reshape(1, num_inputs)})
            obs, reward, done, info = env.step(action_val[0])
            env.render()
            totalreward += reward
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            optimizer.minimize(y-totalreward)
            print(totalreward.values())
           

            
                
print("After {} episodes the average cart steps before done was {}".format(epi,np.mean(avg_steps)))
env.close()
