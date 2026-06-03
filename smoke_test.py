from env import DynamicPricingEnv
env = DynamicPricingEnv()
obs, _ = env.reset()
for _ in range(5):
    obs, reward, done, _, _ = env.step(env.action_space.sample())
    env.render()
    if done:
        break