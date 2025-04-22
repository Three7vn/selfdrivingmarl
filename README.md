# selfdrivingmarl
Multi-Agent Reinforcement Learning For Autonomous Vehicles

## Project: MARL-Based Autonomous Driving Coordination in Highway Environments

### Objective
Train multiple self-driving agents in a shared highway environment to coordinate navigation, maintain safe distances, and avoid collisions.

### Environment
- Use HighwayEnv Simulator (lightweight, Mac-compatible alternative to CARLA)
- Multi-Agent Training: 2+ self-driving agents 
- Configurable traffic density and behavior

### Implementation Steps
1. Setup HighwayEnv for Multi-Agent Simulation
2. Observation Engineering: Define meaningful state representations
3. Reinforcement Learning: Train multi-agent policy gradient algorithm
4. Coordination & Communication: Implement agent-to-agent awareness
5. Testing in Stochastic Environments: Introduce random traffic patterns and behaviors

### Project Structure
```
├── agents/
│   └── pg_agent.py           # Policy gradient agent class (shared or separate)
├── highway_env_wrapper.py    # Multi-agent HighwayEnv interface (step, reset, get_obs)
├── train.py                  # Main training script
├── config.yaml               # Configs (learning rate, episodes, seed, etc.)
├── utils.py                  # Logging, model saving, reward tracking
├── models/                   # Saved model weights
│   └── agent1.pth
│   └── agent2.pth
├── logs/                     # TensorBoard logs or evaluation outputs
├── README.md                 # Basic usage, environment notes
└── requirements.txt          # Python dependencies
```

#### Key Components

**pg_agent.py**  
Stores policy logic; just one file for both agents (parameterized)

**highway_env_wrapper.py**  
Lightweight wrapper to manage multi-agent highway simulation (reset, obs, step, reward)

**train.py**  
Controls training loop, environment interaction, agent updates

**config.yaml**  
Simple hyperparameter management—no hardcoded junk

**utils.py**  
Handles logging, saving, maybe moving averages or reward smoothing

**models/**  
Keeps saved checkpoints per agent (optional eval script later)

**logs/**  
Stores training plots, TensorBoard if needed

**README.md**  
Explains how to install, run, and modify things

**requirements.txt**  
Keeps environment reproducible

### Theoretical Underpinnings
- Multi-Agent MMDP Framework 
- Policy Gradient Optimization 
- Sensor Fusion
- Epsilon-Greedy Exploration 
- Partial Observability Handling 
- Feature Engineering for Highway Scenarios

### Installation
```
# Clone repository
git clone https://github.com/yourusername/selfdrivingmarl.git
cd selfdrivingmarl

# Install dependencies
pip install -r requirements.txt
```

### Usage
```
# Run training
python train.py

# Run with custom config
python train.py --config custom_config.yaml
```

### Advantages of HighwayEnv
- Lightweight and Mac-compatible
- Fast simulation speed enables more training iterations
- Built on Gym interface for easy integration with RL libraries
- Configurable environment complexity
- Support for both continuous and discrete action spaces
