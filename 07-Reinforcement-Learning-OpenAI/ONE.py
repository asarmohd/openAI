
import gym

# Make the environment, replace this string with any
# from the docs. (Some environments have dependencies)
env = gym.make("MountainCarContinuous-v0")
observation = env.reset()

for t in range(1000):

    env.render()


    # Move Cart Right if Pole is Falling to the Right

    # Angle is measured off straight vertical line
    if t < 600:
        # Move Right
        action = 0.1
    else:
        # Move Left
        action = 0.9

    # Perform Action
    action = env.action_space.sample()
    observation , reward, done, info = env.step([0.9])
    print(observation , reward, done, info,action)

