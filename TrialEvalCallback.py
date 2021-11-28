from typing import Optional, Union

import optuna
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.monitor import Monitor


class TrialEvalCallback(EvalCallback):
    """
        stable_baselines3 callback used to evaluate and report a trial.
    """

    def __init__(self,
                 eval_env: Union[VecEnv, Monitor],
                 trial: optuna.Trial,
                 n_eval_episodes: int = 5,
                 eval_freq: int = 10000,
                 deterministic: bool = True,
                 verbose: int = 0,
                 best_model_save_path: Optional[str] = None,
                 log_path: Optional[str] = None,
                 ):
        super(TrialEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
        )
        self.trial = trial
        self.eval_index = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(TrialEvalCallback, self)._on_step()
            self.eval_index += 1
            self.trial.report(self.last_mean_reward, self.eval_index)
            # Prune trial if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
