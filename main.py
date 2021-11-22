import gym
from gym import Env
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure


def test_agent(model, env: Env):
    monitored_env = Monitor(env)
    model.set_env(monitored_env)
    return evaluate_policy(model, monitored_env, n_eval_episodes=10)


def run_cartpole():
    print("Running stable_baselines3 PPO agent learning")
    # Init logger
    logger = configure(None, ["stdout"])

    # Parallel Environments
    env = make_vec_env("CartPole-v1", n_envs=2)

    model = A2C("MlpPolicy", env, learning_rate=0.1, gamma=0.6)
    model.set_logger(logger)
    model.learn(total_timesteps=25000)
    # model.save("models/ppo_cartpole")

    # del model
    # del env
    #
    # model = PPO.load("models/ppo_cartpole")

    env = gym.make("CartPole-v1")
    model.set_env(env)
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

        if done:
            break

    env.close()

    env = gym.make("CartPole-v1")
    model.set_env(env)
    mean_rewards, std_rewards = test_agent(model, env)
    print(f"Performance: Rewards: {mean_rewards}, STD: {std_rewards}")


def main():
    run_cartpole()


if __name__ == "__main__":
    main()
    print("Finished program")
