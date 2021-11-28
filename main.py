import os
import time
from datetime import timedelta
from functools import reduce
import wakepy

import gym
from gym import Env
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure, Logger
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import TrialEvalCallback


def clear_console():
    """
    Clears the console's display.

    :return: None
    """
    import os
    clear_command: str = "cls"
    if os.name != "nt":
        clear_command = "clear"
    os.system(clear_command)


def factors(n):
    """
    Gets a list of the factors for a given number.
    :param n: The number to get factors for.
    :return: A list of all the factors for a given number.
    """
    return set(reduce(list.__add__,
                      ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))


def test_agent(model, env: Env):
    monitored_env = Monitor(env)
    model.set_env(monitored_env)
    return evaluate_policy(model, monitored_env, n_eval_episodes=10)


def run_cart_pole():
    print("Running stable_baselines3 A2C agent learning of cart pole")
    # Init logger
    logger = configure(None, ["stdout"])

    # Parallel Environments
    env = make_vec_env("CartPole-v1", n_envs=3)

    model = A2C("MlpPolicy", env)
    model.set_logger(logger)
    with wakepy.keepawake():
        model.learn(total_timesteps=25000)
    # model.save("models/a2c_cart_pole")

    # del model
    # del env
    #
    # model = PPO.load("models/a2c_cart_pole")

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


def taxi_objective(trial: optuna.Trial):
    """
    Function to use with optuna to tune hyperparameters for taxi environment
    :param trial: The optuna trial
    :return: The mean reward for the trial
    """
    # Create the gym model
    eval_env = Monitor(gym.make("Taxi-v3"))

    # Determine the hyperparameters
    algorithm = trial.suggest_categorical("algorithm", ["PPO", "A2C", "DQN"])
    policy = "MlpPolicy"

    # Get trial's hyperparameters that are common to all algorithms
    learning_rate = trial.suggest_float("learning_rate", 0, 1)
    gamma = trial.suggest_float("gamma", 0, 1)

    if algorithm == "PPO":
        # Get trial's hyperparameters that are for PPO algorithm only
        n_steps = trial.suggest_int("n_steps", 2, 2048 * 5)
        n_epochs = trial.suggest_int("n_epochs", 1, 10 * 5)

        # Suggestion: factors of n_steps * n_envs (number of environments (parallel))
        # batch_size = trial.suggest_categorical("batch_size", factors(n_steps))
        n_steps_factors = factors(n_steps)
        n_steps_factors_copy = n_steps_factors.copy()  # Copy for debugging
        if len(n_steps_factors) > 2:
            n_steps_factors.pop()
        batch_size = n_steps_factors.pop()  # Get second largest factor (or last factor if only two)

        if batch_size == 1:
            print("Invalid batch_size would have been picked")
            print(f"Factors of {n_steps} were: {n_steps_factors_copy}")
            batch_size = n_steps

        model = PPO(policy, eval_env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps, n_epochs=n_epochs,
                    batch_size=batch_size)
    elif algorithm == "A2C":
        # Get trial's hyperparameters that are for PPO algorithm only
        n_steps = trial.suggest_int("n_steps", 1, 5 * 5)

        model = A2C(policy, eval_env, learning_rate=learning_rate, gamma=gamma, n_steps=n_steps)
    elif algorithm == "DQN":
        # batch_size = trial.suggest_int("batch_size", 1, 32 * 5)

        model = DQN(policy, eval_env, learning_rate=learning_rate, gamma=gamma)
    else:
        raise ValueError(f"Invalid algorithm selected: {algorithm}")

    eval_callback = TrialEvalCallback.TrialEvalCallback(eval_env, trial)

    try:
        # No keep awake needed here as this is called by optuna which has been kept awake
        model.learn(25000 * 10, callback=eval_callback)

        model.env.close()
        eval_env.close()
    except (AssertionError, ValueError) as e:
        # Sometimes, random hyperparameters can generate NaN
        model.env.close()
        eval_env.close()
        # Prune hyperparameters that generate NaNs
        print(e)
        print("============")
        print("Sampled hyperparameters:")
        print(trial.params)
        raise optuna.exceptions.TrialPruned()

    is_pruned = eval_callback.is_pruned
    reward = eval_callback.last_mean_reward

    del model.env, eval_callback
    del model

    if is_pruned:
        raise optuna.exceptions.TrialPruned()

    return reward


def perform_optuna_optimizing():
    print("Starting a optuna hyperparameter optimization study run")

    study = optuna.create_study(direction="maximize")

    n_trials = 100

    try:
        with wakepy.keepawake():
            study.optimize(taxi_objective, n_trials=n_trials)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    report_name = (
        f"report_Taxi_{n_trials}-trials-{25000}"
        f"-TPE-None_{int(time.time())}"
    )

    log_path = os.path.join("models", "Taxi", report_name)

    print(f"Writing report to {log_path}")

    # Write report
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    study.trials_dataframe().to_csv(f"{log_path}.csv")

    # Plot optimization result
    try:
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)

        fig1.show()
        fig2.show()
    except (ValueError, ImportError, RuntimeError):
        pass


def perform_taxi_training(logger: Logger):
    # Parallel Environments
    env = make_vec_env("Taxi-v3", n_envs=4)

    try:
        model = PPO.load("models/best_model", env)
    except FileNotFoundError:
        model = None
        logger.log("No existing model found at models/best_model.zip")

    if model is None:
        logger.log("No existing model. Creating a new model to learn with")
        model = PPO("MlpPolicy", env)
    else:
        logger.log("Existing model found. Will continue its learning")

    model.set_logger(logger)

    # Set callbacks
    # Separate evaluation environment
    eval_env = Monitor(gym.make('Taxi-v3'))
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=10, verbose=1)
    eval_callback = EvalCallback(eval_env=eval_env, callback_on_new_best=callback_on_best,
                                 best_model_save_path="models/", verbose=1)
    with wakepy.keepawake():
        model.learn(total_timesteps=25000, callback=eval_callback)
    # model.save("models/ppo_taxi")
    env.close()
    logger.log("Training complete")

    return model


# noinspection PyPep8Naming
def run_taxi():
    print("Running stable_baselines3 PPO agent learning of taxi")

    PERFORM_TRAINING = True
    DISPLAY_GAME = True
    # Init logger
    logger = configure(None, ["stdout"])

    if not PERFORM_TRAINING and not DISPLAY_GAME:
        logger.log("No action to perform, please update script/program and run again")
        return

    if PERFORM_TRAINING:
        model = perform_taxi_training(logger)
    else:
        logger.log("Training option disabled. Loading model from file")
        env = gym.make("Taxi-v3")
        model = PPO.load("models/best_model", env)

    if DISPLAY_GAME:
        env = gym.make("Taxi-v3")
        model.set_env(env)
        state = env.reset()
        done = False

        reward_score = 0
        steps = 0
        for _ in range(1000):
            action, _states = model.predict(state)
            new_state, reward, done, info = env.step(action)
            state = new_state

            reward_score += reward
            steps += 1

            clear_console()
            env.render()
            logger.log(f"Reward: {reward}, Step#: {steps}")
            time.sleep(0.6)

            if done:
                logger.log(f"Reward Sum/Score: {reward_score}, Steps: {steps}")
                break

        if not done:
            logger.log("Agent could not complete the game")

        env.close()
        logger.log("Game complete")

    env = gym.make("Taxi-v3")
    model.set_env(env)
    mean_rewards, std_rewards = test_agent(model, env)
    logger.log(f"Performance: Rewards: {mean_rewards} +/- {std_rewards:.2f}")


def main():
    start_time = time.time()
    # run_taxi()
    perform_optuna_optimizing()
    print(f"Finished program. Execution time: {timedelta(seconds=(time.time() - start_time))}")


if __name__ == "__main__":
    main()
