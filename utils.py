import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

def setup_logger(log_dir):
    """
    Set up logger for experiment.
    
    Args:
        log_dir: Directory to save logs
        
    Returns:
        logger: Logger instance
    """
    log_file = os.path.join(log_dir, 'experiment.log')
    
    # Create logger
    logger = logging.getLogger('marl_agent')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def log_metrics(metrics, step, log_dir):
    """
    Log metrics to TensorBoard.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current step/episode
        log_dir: Directory to save TensorBoard logs
    """
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Log each metric
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)
    
    writer.close()

def save_metrics(metrics, file_path):
    """
    Save metrics to file.
    
    Args:
        metrics: Dictionary of metrics to save
        file_path: Path to save file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(metrics, f)

def load_metrics(file_path):
    """
    Load metrics from file.
    
    Args:
        file_path: Path to metrics file
        
    Returns:
        metrics: Dictionary of metrics
    """
    with open(file_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics

def plot_learning_curve(rewards, losses, save_path):
    """
    Plot learning curves and save to file.
    
    Args:
        rewards: List of rewards for each episode
        losses: List of losses for each episode
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot rewards
    ax1.plot(np.arange(len(rewards)), rewards)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward per Episode')
    ax1.grid(True)
    
    # Plot smoothed rewards
    if len(rewards) > 10:
        smoothed_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
        ax1.plot(np.arange(len(smoothed_rewards)) + 9, smoothed_rewards, 'r-', alpha=0.5)
        ax1.legend(['Rewards', 'Smoothed Rewards'])
    
    # Plot losses
    ax2.plot(np.arange(len(losses)), losses)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss per Episode')
    ax2.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def moving_average(data, window_size):
    """
    Calculate the moving average of a data series.
    
    Args:
        data: List or numpy array of data
        window_size: Window size for moving average
        
    Returns:
        ma: Moving average as numpy array
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def normalize_observation(obs, low, high):
    """
    Normalize observation to range [-1, 1].
    
    Args:
        obs: Observation
        low: Lower bound of observation
        high: Upper bound of observation
        
    Returns:
        normalized_obs: Normalized observation
    """
    range = high - low
    return 2 * (obs - low) / range - 1

def calculate_distance(vehicle1_pos, vehicle2_pos):
    """
    Calculate Euclidean distance between two vehicles in HighwayEnv.
    
    Args:
        vehicle1_pos: Position of first vehicle [x, y]
        vehicle2_pos: Position of second vehicle [x, y]
        
    Returns:
        distance: Euclidean distance between vehicles
    """
    return np.sqrt((vehicle1_pos[0] - vehicle2_pos[0])**2 + 
                  (vehicle1_pos[1] - vehicle2_pos[1])**2)

def plot_multi_agent_rewards(rewards_list, agent_names=None, save_path=None):
    """
    Plot rewards for multiple agents.
    
    Args:
        rewards_list: List of reward lists, one per agent
        agent_names: List of agent names (optional)
        save_path: Path to save plot (optional)
    """
    if agent_names is None:
        agent_names = [f"Agent {i}" for i in range(len(rewards_list))]
    
    plt.figure(figsize=(10, 6))
    
    for i, rewards in enumerate(rewards_list):
        plt.plot(rewards, label=agent_names[i])
        
        # Plot smoothed rewards
        if len(rewards) > 10:
            smoothed_rewards = moving_average(rewards, 10)
            plt.plot(np.arange(len(smoothed_rewards)) + 9, 
                    smoothed_rewards, '--', alpha=0.5)
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Agent')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()
