from ...pushing_envs.pushing_env_lstm_categorical import PushingEnv

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

clip_range = 0.2
epochs = 10
sde = False
target_kl = 0.01
batch_size = 7680
horizon = 300
learning_rate = 0.0003
n_envs = 128

model_name = "ppo_lstm_categorical_part_1"

env = make_vec_env(PushingEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))

checkpoint_callback = CheckpointCallback(
    save_freq=78_125,
    save_path=f"./checkpoints/"+f"{model_name}/",
    name_prefix="checkpoint"
)

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.already_halfed = False

    def _on_step(self):
        if (not self.already_halfed) and (sum(self.training_env.get_attr("CURRENT_SUCCESS_RATE"))/n_envs > 90):
            self.training_env.env_method("half_success_threshold")
            self.already_halfed = True
        self.logger.record("curriculum_step", sum(self.training_env.get_attr("CURRICULUM_STEP"))/n_envs)
        self.logger.record("avg_success_rate", sum(self.training_env.get_attr("CURRENT_SUCCESS_RATE"))/n_envs)
        return True

callback_list = CallbackList([checkpoint_callback, CurriculumCallback()])

class CustomMlpFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim = 128):
        super(CustomMlpFeatureExtractor, self).__init__(observation_space, features_dim)
        self.mlp = nn.Sequential(nn.Linear(8, features_dim), nn.Tanh())

    def forward(self, observations):
        return self.mlp(observations)

model = RecurrentPPO(
    policy="MlpLstmPolicy",
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
    policy_kwargs=dict(features_extractor_class = CustomMlpFeatureExtractor,
                       share_features_extractor = False,    
                       lstm_hidden_size = 256, 
                       n_lstm_layers = 1,
                       net_arch = dict(pi = [128], vf = [128])))
  
model.learn(
    total_timesteps=4e9,
    callback=callback_list,
    tb_log_name=model_name)
