from ....pushing_envs.simplified_pushing_task.pushing_env_mlp_gaussian import PushingEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback

clip_range = 0.2
epochs = 10
sde = False
target_kl = 0.01
batch_size = 7680
horizon = 300
learning_rate = 0.0003
n_envs = 128

policy_arch = [512, 512]
value_arch = [1024, 1024]

model_name = "simplified_task_ppo_mlp_gaussian"
        
env = make_vec_env(PushingEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))

checkpoint_callback = CheckpointCallback(
    save_freq=78_125,
    save_path=f"./checkpoints/"+f"{model_name}/",
    name_prefix="checkpoint"
)

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.steps_after_last_update = 0

    def _on_step(self):
        self.steps_after_last_update += 1
        if (sum(self.training_env.get_attr("CURRENT_SUCCESS_RATE"))/n_envs > 90) and (self.steps_after_last_update > 10_000):
            self.training_env.env_method("curriculum_take_step")
            self.steps_after_last_update = 0
        self.logger.record("curriculum_step", sum(self.training_env.get_attr("CURRICULUM_STEP"))/n_envs)
        self.logger.record("avg_success_rate", sum(self.training_env.get_attr("CURRENT_SUCCESS_RATE"))/n_envs)
        return True

callback_list = CallbackList([checkpoint_callback, CurriculumCallback()])

model = PPO(
    policy="MlpPolicy",
    env = env,
    learning_rate=learning_rate,
    n_steps=horizon,
    batch_size=batch_size,
    n_epochs=epochs,
    clip_range=clip_range,
    use_sde=sde,
    target_kl=target_kl,
    verbose=0, 
    tensorboard_log=f"./tensorboard/",
    policy_kwargs=dict(net_arch=dict(pi=policy_arch, vf=value_arch)))
    
model.learn(
    total_timesteps=5e8,
    callback=callback_list,
    tb_log_name=model_name)
