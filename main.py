import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def run_cartpole():
    print("Running stable_baselines3 PPO agent learning")
    # Parallel Environments
    env = make_vec_env("CartPole-v1", n_envs=2)

    model = PPO("MlpPolicy", env, learning_rate=0.1, gamma=0.6)
    model.learn(total_timesteps=25000)
    model.save("models/ppo_cartpole")

    del model
    del env

    model = PPO.load("models/ppo_cartpole")

    env = gym.make("CartPole-v1")
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

        if done:
            break

    env.close()


def main():
    run_cartpole()


if __name__ == "__main__":
    main()
    print("Finished program")
