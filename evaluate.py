import os
import argparse
import time
import numpy as np
import torch
from agents.pg_agent import PGAgent
from highway_env_wrapper import MultiAgentHighwayEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained MARL agents in roundabout environment")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory containing trained model weights')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to evaluate')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between frames for visualization')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--record', action='store_true', help='Record a video of the evaluation')
    return parser.parse_args()

def evaluate(args):
    """
    Evaluate trained agents in the environment.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Create environment
    env = MultiAgentHighwayEnv(args.config)
    
    # Create agents
    num_agents = env.num_agents
    agents = []
    
    # Load trained agents
    for i in range(num_agents):
        agent = PGAgent(i, args.config)
        model_path = os.path.join(args.model_dir, f"agent{i}.pth")
        
        if os.path.exists(model_path):
            print(f"Loading model for agent {i} from {model_path}")
            agent.load(model_path)
        else:
            print(f"Warning: Model not found for agent {i}, using untrained agent")
        
        agents.append(agent)
    
    # Track episode statistics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0
    
    # Run evaluation episodes
    for episode in range(args.episodes):
        print(f"\nStarting episode {episode+1}/{args.episodes}")
        observations = env.reset()
        episode_reward = [0] * num_agents
        episode_step = 0
        episode_done = False
        
        while not episode_done:
            # Get actions from all agents
            actions = []
            
            for i, agent in enumerate(agents):
                action, _ = agent.get_action(observations[i], training=False)
                actions.append(action)
            
            # Take step in environment
            next_observations, rewards, dones, truncateds, infos = env.step(actions)
            
            # Update episode statistics
            for i in range(num_agents):
                episode_reward[i] += rewards[i]
            
            # Update observations
            observations = next_observations
            
            # Render and wait
            env.render()
            time.sleep(args.delay)
            
            # Check if episode is done
            episode_done = any(dones) or any(truncateds)
            episode_step += 1
            
            # Print step information
            if episode_step % 10 == 0:
                print(f"Step {episode_step}, Rewards: {rewards}")
        
        # Episode complete
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_step)
        
        # Check for success or collision
        if any("crashed" in str(info) for info in infos):
            collision_count += 1
            print(f"Episode {episode+1} ended with collision after {episode_step} steps")
        elif any("success" in str(info) and info.get("success", False) for info in infos):
            success_count += 1
            print(f"Episode {episode+1} ended with success after {episode_step} steps")
        else:
            print(f"Episode {episode+1} ended after {episode_step} steps")
        
        print(f"Episode rewards: {episode_reward}")
    
    # Print overall statistics
    print("\n===== Evaluation Results =====")
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {success_count/args.episodes:.2f}")
    print(f"Collision rate: {collision_count/args.episodes:.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f}")
    
    for i in range(num_agents):
        agent_rewards = [rewards[i] for rewards in episode_rewards]
        print(f"Agent {i} average reward: {np.mean(agent_rewards):.2f}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args) 