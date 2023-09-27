from ...pushing_envs.pushing_env_mlp_gaussian import PushingEnv

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import BaseCallback
import glob

def get_last_model():
    files = glob.glob(f'checkpoints/sac_mlp_gaussian_part_*/*')
    models = []
    for file in files:
        folders = file.split("/")
        part = int(folders[1].split("_")[4])
        checkpoint = int(folders[2].split("_")[-2])
        models.append((part, checkpoint, file))
        
    models = sorted(models, key = lambda x: (x[0], x[1]))

    last_model = models[-1]

    return last_model

last_model_part, last_model_checkpoint, last_model_file = get_last_model()

model_name = f"sac_mlp_gaussian_part_{last_model_part+1}_{last_model_checkpoint}/"

n_envs = 128

env = make_vec_env(PushingEnv, n_envs=n_envs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))

checkpoint_callback = CheckpointCallback(
    save_freq=78_125,
    save_path=f"./checkpoints/"+f"{model_name}/",
    name_prefix="checkpoint"
)

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        self.logger.record("avg_success_rate", sum(self.training_env.get_attr("CURRENT_SUCCESS_RATE"))/n_envs)
        return True

callback_list = CallbackList([checkpoint_callback, CurriculumCallback()])

model = SAC.load(last_model_file, tensorboard_log=f"./tensorboard/")
    
model.set_env(env)

model.learn(
    total_timesteps=4e9,
    callback=callback_list,
    tb_log_name=model_name)
