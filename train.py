import numpy as np
import yaml
import os
import time
import argparse
from datetime import datetime
from agents.pg_agent import PGAgent
from highway_env_wrapper import MultiAgentHighwayEnv
from utils import setup_logger, log_metrics, save_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train MARL agents in Highway Environment")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--shared', action='store_true', help='Use parameter sharing between agents')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--eval', action='store_true', help='Run in evaluation mode (no training)')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--delay', type=float, default=0.2, help='Rendering delay in seconds')
    parser.add_argument('--smooth', action='store_true', help='Use smoother control for visualization')
    return parser.parse_args()

def train(config_path, shared_parameters=False, resume_path=None, eval_mode=False, render=False, delay=0.2, smooth=False):
    """
    Train multiple agents in the Highway environment.
    
    Args:
        config_path: Path to configuration file
        shared_parameters: Whether to share parameters between agents
        resume_path: Path to checkpoint to resume from
        eval_mode: Whether to run in evaluation mode
        render: Whether to render the environment
        delay: Delay between frames when rendering
        smooth: Use smoother control for visualization
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join("logs", f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logger(experiment_dir)
    logger.info(f"Starting experiment with config: {config}")
    
    # Create environment
    env = MultiAgentHighwayEnv(config_path)
    
    # Create agents
    num_agents = config['agents']['num_agents']
    agents = []
    
    # Updated state dimension after sensor fusion
    from agents.pg_agent import PGAgent
    
    # For parameter sharing, create a network that will be shared
    shared_network = None
    if shared_parameters:
        logger.info("Using parameter sharing between agents")
        # Create first agent
        first_agent = PGAgent(0, config_path)
        # Get its network
        shared_network = first_agent.policy_network
        agents.append(first_agent)
        
        # Create other agents with shared network
        for i in range(1, num_agents):
            agents.append(PGAgent(i, config_path, shared_network=shared_network))
    else:
        # Create independent agents
        for i in range(num_agents):
            agents.append(PGAgent(i, config_path))
    
    # Resume from checkpoint if specified
    if resume_path:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        # Load weights for all agents
        for i, agent in enumerate(agents):
            agent_path = os.path.join(resume_path, f"agent{i}.pth")
            if os.path.exists(agent_path):
                agent.load(agent_path)
    
    # Training configuration
    num_epochs = config['training']['epochs']
    episodes_per_epoch = config['training']['episodes_per_epoch']
    log_interval = config['logging']['log_interval']
    checkpoint_interval = config['logging']['checkpoint_interval']
    
    # Training loop
    episode_rewards = [[] for _ in range(num_agents)]
    losses = []
    
    try:
        for epoch in range(num_epochs):
            epoch_rewards = [[] for _ in range(num_agents)]
            epoch_losses = []
            
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            
            for episode in range(episodes_per_epoch):
                # Reset environment and get initial observations
                observations = env.reset()
                episode_reward = [0] * num_agents
                episode_done = False
                episode_step = 0
                
                logger.info(f"Starting episode {episode+1}/{episodes_per_epoch}")
                
                # Run episode
                while not episode_done:
                    # Get actions from all agents
                    actions = []
                    action_log_probs = []
                    
                    # Use smoother control for visualization (more IDLE actions)
                    if smooth and (render or eval_mode) and np.random.random() < 0.3:
                        # 30% chance of IDLE for smoother visualization
                        for i in range(num_agents):
                            # Use IDLE action (index may vary based on action space)
                            idle_action = 1  # Usually the IDLE action
                            action_prob = -1.0  # Placeholder
                            actions.append(idle_action)
                            action_log_probs.append(action_prob)
                    else:
                        # Normal action selection
                        for i, agent in enumerate(agents):
                            action, action_log_prob = agent.get_action(observations[i], training=not eval_mode)
                            actions.append(action)
                            action_log_probs.append(action_log_prob)
                    
                    # Take step in environment
                    next_observations, rewards, dones, truncateds, infos = env.step(actions)
                    
                    # Store rewards
                    for i in range(num_agents):
                        episode_reward[i] += rewards[i]
                    
                    # Store experiences
                    if not eval_mode:
                        for i, agent in enumerate(agents):
                            agent.remember(
                                observations[i], 
                                actions[i], 
                                action_log_probs[i], 
                                rewards[i], 
                                next_observations[i], 
                                dones[i]
                            )
                    
                    # Update observations
                    observations = next_observations
                    
                    # Check if episode is done (any agent is done or truncated)
                    episode_done = any(dones) or any(truncateds) or episode_step >= config['environment']['max_episode_steps']
                    episode_step += 1
                    
                    # Render if in evaluation mode or if render flag is set
                    if render or eval_mode:
                        env.render()
                        time.sleep(delay)  # Add delay for visualization
                        
                        # Print step info occasionally
                        if episode_step % 10 == 0:
                            logger.info(f"Step {episode_step}, Rewards: {rewards}")
                
                # Add episode rewards to epoch rewards
                for i in range(num_agents):
                    epoch_rewards[i].append(episode_reward[i])
                
                # Train agents after episode
                if not eval_mode:
                    for i, agent in enumerate(agents):
                        loss = agent.train()
                        epoch_losses.append(loss)
                
                # Log episode results
                if episode % log_interval == 0 or episode == episodes_per_epoch - 1:
                    logger.info(f"Epoch {epoch}, Episode {episode}: Rewards={episode_reward}")
            
            # Calculate epoch metrics
            mean_rewards = [np.mean(rew) for rew in epoch_rewards]
            std_rewards = [np.std(rew) for rew in epoch_rewards]
            mean_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            
            # Log epoch results
            logger.info(f"Epoch {epoch}: Mean Rewards={mean_rewards}, Mean Loss={mean_loss}")
            
            # Add to total metrics
            for i in range(num_agents):
                episode_rewards[i].extend(epoch_rewards[i])
            losses.extend(epoch_losses)
            
            # Save metrics
            metrics = {
                'episode_rewards': episode_rewards,
                'losses': losses,
                'epochs': epoch + 1
            }
            save_metrics(metrics, os.path.join(experiment_dir, 'metrics.pkl'))
            
            # Save models at checkpoint intervals
            if (epoch + 1) % checkpoint_interval == 0 or epoch == num_epochs - 1:
                for i, agent in enumerate(agents):
                    agent.save(os.path.join(experiment_dir, f"agent{i}.pth"))
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Clean up
        env.close()
        logger.info("Training complete")

if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.shared, args.resume, args.eval, args.render, args.delay, args.smooth)
