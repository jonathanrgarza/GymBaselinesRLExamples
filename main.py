import time
import gym
from gym import Env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold


def clear_console():
    """
    Clears the console's display.

    :return: None
    """
    import os
    clear_command: str = "cls"
    if os.name != "ns":
        clear_command = "clear"
    os.system(clear_command)


def test_agent(model, env: Env):
    monitored_env = Monitor(env)
    model.set_env(monitored_env)
    return evaluate_policy(model, monitored_env, n_eval_episodes=10)


def run_cartpole():
    print("Running stable_baselines3 A2C agent learning of cartpole")
    # Init logger
    logger = configure(None, ["stdout"])

    # Parallel Environments
    env = make_vec_env("CartPole-v1", n_envs=3)

    model = A2C("MlpPolicy", env)
    model.set_logger(logger)
    model.learn(total_timesteps=25000)
    # model.save("models/a2c_cartpole")

    # del model
    # del env
    #
    # model = PPO.load("models/a2c_cartpole")

    env = gym.make("CartPole-v1")
    model.set_env(env)
    state = env.reset()
    for _ in range(1000):
        action, _states = model.predict(state)
        new_state, reward, done, info = env.step(action)
        state = new_state

        print(f"Reward: {reward}")
        env.render()

        if done:
            break

    env.close()

    env = gym.make("CartPole-v1")
    model.set_env(env)
    mean_rewards, std_rewards = test_agent(model, env)
    print(f"Performance: Rewards: {mean_rewards} +/- {std_rewards:.2f}")


def run_taxi():
    print("Running stable_baselines3 PPO agent learning of taxi")

    DISPLAY_GAME = False
    # Init logger
    logger = configure(None, ["stdout"])

    # Parallel Environments
    env = make_vec_env("Taxi-v3", n_envs=4)

    model = PPO.load("models/best_model.zip", env)
    if model is None:
        print("No existing model. Creating a new model to learn with")
        model = PPO("MlpPolicy", env)
    else:
        print("Existing model found. Will continue its learning")

    model.set_logger(logger)

    # Set callbacks
    # Separate evaluation environment
    eval_env = Monitor(gym.make('Taxi-v3'))
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=10, verbose=1)
    eval_callback = EvalCallback(eval_env=eval_env, callback_on_new_best=callback_on_best,
                                 best_model_save_path="models/", verbose=1)

    model.learn(total_timesteps=25000 * 100, callback=eval_callback)
    # model.save("models/ppo_taxi")
    env.close()

    # del model
    # del env
    #
    # model = PPO.load("models/ppo_taxi")
    if DISPLAY_GAME:
        env = gym.make("Taxi-v3")
        model.set_env(env)
        state = env.reset()
        done = False
        for _ in range(1000):
            action, _states = model.predict(state)
            new_state, reward, done, info = env.step(action)
            state = new_state

            clear_console()
            env.render()
            print(f"Reward: {reward}")
            time.sleep(0.6)

            if done:
                break

        if not done:
            print("Agent could not complete the game")

        env.close()

    env = gym.make("Taxi-v3")
    model.set_env(env)
    mean_rewards, std_rewards = test_agent(model, env)
    print(f"Performance: Rewards: {mean_rewards} +/- {std_rewards:.2f}")


def main():
    run_taxi()


if __name__ == "__main__":
    main()
    print("Finished program")
