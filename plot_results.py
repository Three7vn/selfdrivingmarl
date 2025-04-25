import os
import pickle
from utils import plot_multi_agent_rewards

# Find latest experiment directory
dirs = [d for d in os.listdir('logs') if d.startswith('experiment_')]
latest = sorted(dirs)[-1]
metrics_path = os.path.join('logs', latest, 'metrics.pkl')
metrics = pickle.load(open(metrics_path, 'rb'))

# Extract rewards and plot
episode_rewards = metrics['episode_rewards']  # List for each agent
agent_names = [f"Agent {i+1}" for i in range(len(episode_rewards))]
output_path = os.path.join('logs', latest, 'results_new.png')
plot_multi_agent_rewards(episode_rewards, agent_names, save_path=output_path)
print(f"Plot saved to {output_path}") 