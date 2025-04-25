import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import yaml

class PolicyNetwork(nn.Module):
    """
    Neural network for policy-based RL.
    Takes observations as input and outputs action probabilities.
    """
    def __init__(self, input_dim, output_dim, hidden_dims, activation='relu'):
        super(PolicyNetwork, self).__init__()
        
        # Define activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        else:
            self.activation = F.relu
        
        # Input layer
        layers = [nn.Linear(input_dim, hidden_dims[0])]
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        
        # Output layer - no activation for continuous actions
        # For discrete actions, we would apply softmax outside
        x = self.layers[-1](x)
        
        return x


class PGAgent:
    """
    Policy Gradient agent that can be used in single or multi-agent settings.
    
    Key features:
    - Supports both discrete and continuous action spaces
    - Implements vanilla policy gradient (REINFORCE) algorithm
    - Can be instantiated multiple times for different agents
    - Supports parameter sharing between agents
    """
    
    def __init__(self, agent_id, config_path="config.yaml", shared_network=None):
        """
        Initialize a PG agent.
        
        Args:
            agent_id: Identifier for this agent
            config_path: Path to configuration file
            shared_network: Optional shared policy network (for parameter sharing)
        """
        self.agent_id = agent_id
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.agent_config = self.config['agents']
        self.training_config = self.config['training']
        self.network_config = self.config['network']
        
        # Define action space properties
        self.discrete_actions = self.agent_config['action_space']['discrete']
        if self.discrete_actions:
            # Set action dimension based on discrete actions defined in config
            self.action_dim = len(self.agent_config['action_space']['actions'])
        else:
            # Continuous actions for highway env (typically 2: steering, acceleration)
            self.action_dim = 2
        
        # Determine input dimension from observation space
        self.input_dim = self._calc_input_dim()
        
        # Create policy network
        if shared_network is not None:
            self.policy_network = shared_network
            self.shared_network = True
        else:
            hidden_dims = self.network_config['hidden_layers']
            activation = self.network_config['activation']
            self.policy_network = PolicyNetwork(
                self.input_dim, 
                self.action_dim, 
                hidden_dims, 
                activation
            )
            self.shared_network = False
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=self.training_config['learning_rate']
        )
        
        # Memory for storing episode data
        self.memory = []
        
        # Discount factor
        self.gamma = self.training_config['gamma']
        
        # Exploration strategy
        self.epsilon = self.training_config['epsilon_start']
        self.epsilon_end = self.training_config['epsilon_end']
        self.epsilon_decay = self.training_config['epsilon_decay']
    
    def _calc_input_dim(self):
        """Calculate input dimension based on observation space."""
        # After sensor fusion, each vehicle yields [distance, speed, angle]
        vehicles_count = self.agent_config['observation_space']['vehicles_count']
        fused_features_per_vehicle = 3  # distance, speed, angle
        return vehicles_count * fused_features_per_vehicle
    
    def _preprocess_observation(self, observation):
        """
        Preprocess the raw observation into network input.
        
        For HighwayEnv, observation is typically a numpy array of vehicle features.
        """
        # Perform simple sensor fusion before feeding to the network
        fused = self._sensor_fusion(observation)
        return fused.astype(np.float32)
    
    def _sensor_fusion(self, observation):
        """
        Simple sensor fusion: compute for each vehicle the distance,
        speed magnitude, and angle relative to ego.
        """
        # observation shape: (vehicles_count, 6) with [presence, x, y, vx, vy, heading]
        obs = np.array(observation).reshape(-1, 6)
        fused_feats = []
        for row in obs:
            presence = row[0]
            x, y, vx, vy = row[1], row[2], row[3], row[4]
            # Distance to ego
            dist = np.sqrt(x**2 + y**2) * presence
            # Speed magnitude
            speed = np.sqrt(vx**2 + vy**2) * presence
            # Angle relative to ego heading
            angle = np.arctan2(y, x) * presence
            fused_feats.append([dist, speed, angle])
        return np.array(fused_feats).flatten()
    
    def get_action(self, observation, training=True):
        """
        Select an action based on the current policy and observation.
        
        Args:
            observation: Environment observation
            training: Whether to use exploration or not
        
        Returns:
            action: Selected action
            action_log_prob: Log probability of selected action (for training)
        """
        # Preprocess observation
        state = self._preprocess_observation(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action distribution from policy network
        action_params = self.policy_network(state_tensor)
        
        # Different behavior for discrete vs continuous actions
        if self.discrete_actions:
            # Apply softmax to get probabilities
            action_probs = F.softmax(action_params, dim=1)
            
            # Exploration during training
            if training and np.random.random() < self.epsilon:
                action = np.random.randint(0, self.action_dim)
                action_tensor = torch.tensor([action])
            else:
                # Sample action from probability distribution
                action_distribution = torch.distributions.Categorical(action_probs)
                action_tensor = action_distribution.sample()
                action = action_tensor.item()
            
            # Compute log probability for the selected action
            action_log_prob = torch.log(action_probs[0, action])
            
            return action, action_log_prob
            
        else:  # Continuous actions
            # For continuous control, output is mean of a Gaussian distribution
            # with fixed standard deviation for exploration
            action_means = action_params[0]
            
            # Fixed standard deviation for exploration
            # Higher values encourage more exploration
            action_stddev = 0.5 * torch.ones_like(action_means)
            
            # Reduce exploration if not training
            if not training:
                action_stddev = 0.1 * action_stddev
            
            # Create normal distribution
            normal_dist = torch.distributions.Normal(action_means, action_stddev)
            
            # Sample action
            action_tensor = normal_dist.sample()
            
            # Clip actions to valid range
            action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
            
            # Compute log probability
            action_log_prob = normal_dist.log_prob(action_tensor).sum()
            
            # Convert to numpy array
            action = action_tensor.detach().numpy()
            
            return action, action_log_prob
    
    def remember(self, state, action, action_log_prob, reward, next_state, done):
        """Store experience in memory for training."""
        self.memory.append({
            'state': state,
            'action': action,
            'action_log_prob': action_log_prob,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def train(self):
        """Update policy based on collected experience."""
        if len(self.memory) == 0:
            return 0.0
        
        # Extract data from memory
        states = np.array([exp['state'] for exp in self.memory])
        actions = [exp['action'] for exp in self.memory]
        # Stack log-prob tensors to preserve gradient information
        action_log_probs = torch.stack([exp['action_log_prob'] for exp in self.memory])
        rewards = np.array([exp['reward'] for exp in self.memory])
        dones = np.array([exp['done'] for exp in self.memory])
        
        # Calculate returns
        returns = self._compute_returns(rewards, dones)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Policy gradient loss
        policy_loss = -(action_log_probs * returns).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Update exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Clear memory
        self.memory = []
        
        return policy_loss.item()
    
    def _compute_returns(self, rewards, dones):
        """Compute discounted returns for each step."""
        returns = np.zeros_like(rewards, dtype=np.float32)
        
        # Start from the last step and work backwards
        cumulative_return = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                cumulative_return = 0
            
            cumulative_return = rewards[t] + self.gamma * cumulative_return
            returns[t] = cumulative_return
            
        return returns
    
    def save(self, path):
        """Save model to a file."""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model from a file."""
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
