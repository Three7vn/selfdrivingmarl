import matplotlib.pyplot as plt
from highway_env_wrapper import MultiAgentHighwayEnv
from utils import plot_multi_agent_rewards
from agents.pg_agent import PGAgent

# Initialize environment and run one episode
env = MultiAgentHighwayEnv()
observations = env.reset()
rewards_list = [[] for _ in range(env.num_agents)]

# Create policy agents (untrained or load weights if available)
agents = []
for i in range(env.num_agents):
    pg = PGAgent(i)
    # Try to load pretrained model if available
    try:
        pg.load(f"models/agent{i}.pth")
    except Exception:
        pass
    agents.append(pg)

# Step through the episode
for step in range(env.max_steps):
    # Use policy network to select actions for both agents
    actions = []
    for i, agent in enumerate(agents):
        action, _ = agent.get_action(observations[i], training=False)
        actions.append(action)
    observations, rewards, dones, truncateds, infos = env.step(actions)
    for i, r in enumerate(rewards):
        rewards_list[i].append(r)
    if any(dones):
        break

# Close environment
env.close()

# Plot results and save
plot_multi_agent_rewards(
    rewards_list,
    agent_names=[f"Agent {i+1}" for i in range(env.num_agents)],
    save_path="accurate_results.png"
)
print("Saved accurate_results.png") 