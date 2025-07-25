import gym
import os
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
import argparse
from src.env_wrappers import EnvironmentWrappers, DiscreteCarEnvironment, RewardWrapper

WIDTH = 84
HEIGHT = 84
STACKED_FRAMES = 4
REPLAY_BUFFER_SIZE = 50000
BATCH_SIZE = 64
TIME_STEPS = 5000000
NUM_EPISODES = 10
LEARNING_STARTS = 5000


def train_model(model_name, buf_size, batch_size, log_path, model_path, num_steps):
    if model_name == "DQN":
        model = DQN('CnnPolicy', env, verbose=1, device='cuda', buffer_size=REPLAY_BUFFER_SIZE, batch_size=BATCH_SIZE,
                    tensorboard_log=log_path, learning_starts=LEARNING_STARTS)
    else:
        if model_name == "PPO":
            model = PPO('CnnPolicy', env, verbose=1, device='cuda', tensorboard_log=log_path)
        else:
            return -1

    save_best_path = os.path.join(model_path, model_name, 'best_model')
    save_final_path = os.path.join(model_path, model_name, 'final_model', 'final_model.zip')

    print(save_final_path)
    print(save_best_path)

    eval_env = model.get_env()
    eval_callback = EvalCallback(eval_env=eval_env, best_model_save_path=save_best_path,
                                 n_eval_episodes=5,
                                 eval_freq=50000, verbose=1,
                                 deterministic=True, render=False)
    model.learn(total_timesteps=num_steps, callback=eval_callback)
    model.save(save_final_path)
    return save_final_path


def resume_train_model(model_name, log_path, model_path, environment, num_steps):
    path = os.path.join(model_path, model_name, "best_model.zip")
    save_best_path = os.path.join(model_path, model_name, "best_model")
    save_final_path = os.path.join(model_path, model_name, "final_model", "final_model.zip")
    print(f"Logging to Tensorboard: {log_path}")
    print(f"Saving models to: {model_path}")

    if model_name == "DQN":
        model = DQN.load(path, tensorboard_log=log_path)
        print(f"Resuming DQN training")
    else:
        if model_name == "PPO":
            model = PPO.load(path, tensorboard_log=log_path)
            print(f"Resuming PPO training")
        else:
            return -1

    model.set_env(environment)
    eval_callback = EvalCallback(eval_env=model.get_env(), best_model_save_path=save_best_path,
                                 n_eval_episodes=5,
                                 eval_freq=50000, verbose=1,
                                 deterministic=True, render=False)
    model.learn(total_timesteps=num_steps, callback=eval_callback, reset_num_timesteps=False)
    model.save(save_final_path)
    return save_final_path


def test_model(environment, model_name, model_path, episodes):
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=episodes, render=True)
    model_path = os.path.join(model_path, model_name, "best_model.zip")
    print(model_path)
    if model_name == "DQN":
        model = DQN.load(model_path, environment)
    else:
        if model_name == "PPO":
            model = PPO.load(model_path, environment)
        else:
            return -1
    # monitor each episode - record all episodes
    from gym.wrappers.record_video import RecordVideo
    environment = RecordVideo(environment, video_folder="monitor", name_prefix="episode")

    model.set_env(environment)
    environment = model.get_env()

    for episode in range(1, episodes + 1):
        obs = environment.reset()
        done = False
        score = 0
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = environment.step(action)
            score += reward
        print("Episode:{} Score:{}".format(episode, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test", help="Choose mode: train/test/resume")
    parser.add_argument("-a", "--agent", default="DQN", help="Reinforcement learning model/agent to use")
    parser.add_argument("-l", "--log_path", default=os.path.join(".", "Training", "Logs"),
                        help="Directory to store logs")
    parser.add_argument("-b", "--buffer_size", default=REPLAY_BUFFER_SIZE, help="Replay buffer size for DQN")
    parser.add_argument("-t", "--time_steps", default=TIME_STEPS, help="Number of training time steps")
    parser.add_argument("-m", "--model_path", default=os.path.join(".", "Training", "Saved_Models"),
                        help="Directory where the models are stored")
    args = parser.parse_args()

    if args.mode == "test" and (args.model_path is None):
        parser.error("--mode=test requires providing valid --model_path")

    env_wrappers = EnvironmentWrappers(WIDTH, HEIGHT, STACKED_FRAMES)

    env = DiscreteCarEnvironment(gym.make("CarRacing-v2", render_mode="rgb_array"))
    funcs = [env_wrappers.resize, env_wrappers.grayscale, env_wrappers.frame_stack]
    env = env_wrappers.observation_wrapper(env, funcs)
    # env = RewardWrapper(env)

    # apply 3 transformation wrappers to DQN model (resize, grayscale, stacking frames)

    print(f"Observation space shape: {env.observation_space.shape}")

    if args.mode == "train":
        saved_path = train_model(args.agent, args.buffer_size, BATCH_SIZE,
                                 args.log_path, args.model_path, args.time_steps)
        print(f'Model successfully trained after {args.time_steps} time steps. Results saved in model '
              f'file {saved_path}')
    else:
        if args.mode == "resume":
            saved_path = resume_train_model(args.agent, args.log_path, args.model_path, env, args.time_steps)
            print(f'Model successfully trained after {args.time_steps} time steps. Results saved in model '
                  f'file {saved_path}')
        if args.mode == "test":
            test_model(env, args.agent, args.model_path, NUM_EPISODES)
        else:
            exit(-1)

    env.close()
