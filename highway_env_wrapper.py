try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        raise ImportError(
            "This module requires gymnasium (>=0.26.3) or gym. "
            "Please install one of them via 'pip install gymnasium' or 'pip install gym'."
        )
import highway_env
import numpy as np
import yaml
from typing import Dict, List, Tuple, Any, Optional
import time

class MultiAgentHighwayEnv:
    """
    A wrapper around HighwayEnv to support multiple agents in a shared environment.
    Each agent controls a different vehicle and receives its own observations and rewards.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the multi-agent highway environment.
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create base environment
        self.env = gym.make(
            self.config['environment']['env_id'],
            render_mode=self.config['environment']['render_mode']
        )
        
        # Configure environment - accessing the unwrapped environment
        env_config = {
            "observation": {
                "type": self.config['environment']['observation_type'],
                "vehicles_count": self.config['agents']['observation_space']['vehicles_count'],
                "features": ["presence", "x", "y", "vx", "vy", "heading"],
                "absolute": False,
                "normalize": True,
                "observation_shape": (self.config['agents']['observation_space']['vehicles_count'], 6)
            },
            "action": {
                "type": "DiscreteMetaAction" if self.config['agents']['action_space']['discrete'] else "ContinuousAction"
            },
            "vehicles_count": self.config['environment']['vehicles_count'],
            "duration": self.config['environment']['max_episode_steps'],
            "collision_reward": self.config['environment'].get('collision_reward', -5.0),
            "high_speed_reward": self.config['environment'].get('high_speed_reward', 0.1),
            "right_lane_reward": self.config['environment'].get('right_lane_reward', 0.3),
            "lane_change_reward": -0.2,  # Penalty for frequent lane changes
            "normalize_reward": self.config['environment'].get('normalize_reward', True),
            "offroad_terminal": False,  # Prevent early termination
            "render_mode": self.config['environment']['render_mode'],
            "destination": None,  # Allow vehicles to drive forever
            
            # Visual parameters
            "screen_width": self.config['environment'].get('screen_width', 900),
            "screen_height": self.config['environment'].get('screen_height', 700),
            "scaling": self.config['environment'].get('scaling', 6.0),
            "centering_position": [0.5, 0.5],
            "show_trajectories": self.config['environment'].get('show_trajectories', True),
            
            # Two-way traffic specific parameters
            "lanes_count": self.config['environment'].get('lanes_count', 4),  # 2 lanes per direction
            "vehicles_density": 0.7,  # Medium density for clearer traffic
            "initial_lane_id": None,  # Let agent spawn in any lane
            "lane_width": self.config['environment'].get('lane_width', 4.0),
            "ego_spacing": 2.0,  # Increased spacing between ego vehicles
            "vehicles_speed": 15,  # Control flow speed
            
            # Vehicle dynamics
            "simulation_frequency": 15,  # Higher simulation frequency for smoother control
            "policy_frequency": 5,       # Lower policy frequency for smoother control
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "spawn_probability": 0.5,    # Medium vehicle spawning rate for oncoming traffic
        }
        self.env.unwrapped.configure(env_config)
        
        # Force initialize the viewer
        obs, info = self.env.reset()
        self.env.render()
        
        # Multi-agent setup
        self.num_agents = self.config['agents']['num_agents']
        self.controlled_vehicles = []
        
        # Define observation and action spaces for each agent
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # For tracking agent positions and visualizing them differently
        self.agent_colors = [(0, 255, 0), (0, 192, 255)]  # Green for agent 1, Light blue for agent 2
        self.agent_names = ["Agent 1", "Agent 2"]
        
        # Keep track of how many steps we've taken in current episode
        self.current_step = 0
        self.max_steps = self.config['environment']['max_episode_steps']
        self.reset_count = 0
        
        # For collision avoidance between agents
        self.previous_actions = [0, 0]
        self.action_repeat_count = [0, 0]
        
        # Prevent constant resets
        self.min_steps_before_reset = 30  # Minimum number of steps before allowing a reset
        self.allow_early_reset = False    # Only allow resets after min_steps_before_reset
        
        # Initialize vehicles
        self._initialize_vehicles()
        
    def _initialize_vehicles(self):
        """Initialize the controlled vehicles."""
        # First get any existing controlled vehicles
        self._update_controlled_vehicles()
        
        # Try to create properly positioned vehicles 
        self._setup_two_way_vehicles()
        
        # Apply visual modifications
        self._modify_controlled_vehicles()
        
    def _modify_controlled_vehicles(self):
        """Make the controlled vehicles more visible."""
        for i, vehicle in enumerate(self.controlled_vehicles[:self.num_agents]):
            if hasattr(vehicle, 'color'):
                vehicle.color = self.agent_colors[i % len(self.agent_colors)]
            if hasattr(vehicle, 'length'):
                vehicle.length = 5.0  # Make the vehicle a bit larger for visibility
            if hasattr(vehicle, 'width'):
                vehicle.width = 2.2   # Make the vehicle a bit wider for visibility
            
            # Adjust dynamics for smoother driving
            if hasattr(vehicle, 'max_speed'):
                vehicle.max_speed = 25  # Slightly higher max speed
                
            # Add speed_index attribute if needed by the environment
            if not hasattr(vehicle, 'speed_index') and hasattr(vehicle, 'speed'):
                # Map speed to index (0=stopped, 1=half speed, 2=full speed)
                max_speed = getattr(vehicle, 'max_speed', 30)
                speed = getattr(vehicle, 'speed', 0)
                speed_index = min(int(speed / max_speed * 3), 2)
                setattr(vehicle, 'speed_index', speed_index)
                
    def _update_controlled_vehicles(self):
        """Update the list of controlled vehicles."""
        # Get controlled vehicles from the environment
        if hasattr(self.env.unwrapped, 'controlled_vehicles') and self.env.unwrapped.controlled_vehicles:
            self.controlled_vehicles = self.env.unwrapped.controlled_vehicles.copy()
        elif hasattr(self.env.unwrapped, 'vehicle') and self.env.unwrapped.vehicle:
            self.controlled_vehicles = [self.env.unwrapped.vehicle]
        else:
            # Find all vehicles in the road
            if hasattr(self.env.unwrapped, 'road') and hasattr(self.env.unwrapped.road, 'vehicles'):
                self.controlled_vehicles = self.env.unwrapped.road.vehicles[:1]  # Just take the first one
            else:
                self.controlled_vehicles = []
    
    def _setup_two_way_vehicles(self):
        """Create properly positioned vehicles for two-way environment."""
        try:
            # Get the road
            if not hasattr(self.env.unwrapped, 'road') or not self.env.unwrapped.road:
                print("Warning: No road found in environment")
                return
                
            road = self.env.unwrapped.road
            
            # Make sure we have the required vehicle classes
            try:
                from highway_env.vehicle.kinematics import Vehicle
                from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
                from highway_env.vehicle.behavior import IDMVehicle
            except ImportError as e:
                print(f"Error importing vehicle classes: {e}")
                return
            
            # Get lane count
            lanes_count = self.config['environment'].get('lanes_count', 4) // 2  # Lanes per direction
            
            # Clear existing controlled vehicles
            if hasattr(self.env.unwrapped, 'controlled_vehicles'):
                self.env.unwrapped.controlled_vehicles = []
            
            # Create our controlled vehicles
            self.controlled_vehicles = []
            
            # Check available roads and lanes
            road_nets = []
            if hasattr(road.network, 'graph'):
                print("Available roads:")
                for origin in road.network.graph:
                    for destination in road.network.graph[origin]:
                        if road.network.graph[origin][destination]:  # If lanes exist
                            lane_count = len(road.network.graph[origin][destination])
                            road_nets.append((origin, destination, lane_count))
                            print(f"  Road: {origin}->{destination}, Lanes: {lane_count}")
                            for lane_id in range(lane_count):
                                try:
                                    lane = road.network.get_lane((origin, destination, lane_id))
                                    if lane:
                                        print(f"    Lane {lane_id}: OK")
                                except Exception as e:
                                    print(f"    Lane {lane_id}: Error - {e}")
            
            if not road_nets:
                print("Warning: No roads found in environment")
                return
            
            # Try to use the env's vehicle class directly if available
            if hasattr(self.env.unwrapped, 'action_type') and hasattr(self.env.unwrapped.action_type, 'vehicle_class'):
                vehicle_class = self.env.unwrapped.action_type.vehicle_class
            else:
                # Fall back to MDPVehicle which has speed_index
                vehicle_class = MDPVehicle
            
            # Get first valid origin-destination pair for forward direction
            forward_road = None
            backward_road = None
            for origin, destination, lane_count in road_nets:
                if lane_count > 0:
                    if not forward_road:
                        forward_road = (origin, destination)
                        print(f"Using forward road: {origin}->{destination}")
                    elif not backward_road and origin != forward_road[0]:
                        backward_road = (origin, destination)
                        print(f"Using backward road: {origin}->{destination}")
                if forward_road and backward_road:
                    break
                
            # If we couldn't find separate roads, use the same road for both directions
            if not forward_road:
                if road_nets:
                    forward_road = (road_nets[0][0], road_nets[0][1])
                    print(f"Using default forward road: {forward_road[0]}->{forward_road[1]}")
                else:
                    print("No valid roads found")
                    return
                
            if not backward_road:
                backward_road = forward_road
                print(f"Using default backward road: {backward_road[0]}->{backward_road[1]}")
            
            # Try to create first agent on forward road
            if forward_road:
                lane_index = (forward_road[0], forward_road[1], 0)  # First lane
                try:
                    lane = road.network.get_lane(lane_index)
                    vehicle = vehicle_class(
                        road=road,
                        position=lane.position(30, 0),
                        heading=lane.heading_at(30),
                        speed=20
                    )
                    road.vehicles.append(vehicle)
                    self.controlled_vehicles.append(vehicle)
                    print(f"Created Agent 1 on {forward_road[0]}->{forward_road[1]}, lane 0")
                except Exception as e:
                    print(f"Error creating first agent: {e}")
                    self._create_fallback_vehicle(road, forward_road, vehicle_class)
            
            # Try to create second agent on backward road
            if backward_road and len(road_nets) > 1:
                lane_index = (backward_road[0], backward_road[1], 0)  # First lane
                try:
                    lane = road.network.get_lane(lane_index)
                    vehicle = vehicle_class(
                        road=road,
                        position=lane.position(30, 0),
                        heading=lane.heading_at(30),
                        speed=20
                    )
                    road.vehicles.append(vehicle)
                    self.controlled_vehicles.append(vehicle)
                    print(f"Created Agent 2 on {backward_road[0]}->{backward_road[1]}, lane 0")
                except Exception as e:
                    print(f"Error creating second agent: {e}")
                    self._create_fallback_vehicle(road, backward_road, vehicle_class)
            else:
                # Create second agent on same road but different lane if possible
                if forward_road and len(road.network.graph[forward_road[0]][forward_road[1]]) > 1:
                    lane_index = (forward_road[0], forward_road[1], 1)  # Second lane
                    try:
                        lane = road.network.get_lane(lane_index)
                        vehicle = vehicle_class(
                            road=road,
                            position=lane.position(60, 0),  # Different position
                            heading=lane.heading_at(60),
                            speed=20
                        )
                        road.vehicles.append(vehicle)
                        self.controlled_vehicles.append(vehicle)
                        print(f"Created Agent 2 on {forward_road[0]}->{forward_road[1]}, lane 1")
                    except Exception as e:
                        print(f"Error creating second agent on different lane: {e}")
                        self._create_fallback_vehicle(road, forward_road, vehicle_class)
                else:
                    # Create on same lane but different position
                    if len(self.controlled_vehicles) > 0:
                        try:
                            lane_index = self.controlled_vehicles[0].lane_index
                            lane = road.network.get_lane(lane_index)
                            vehicle = vehicle_class(
                                road=road,
                                position=lane.position(80, 0),  # Different position
                                heading=lane.heading_at(80),
                                speed=20
                            )
                            road.vehicles.append(vehicle)
                            self.controlled_vehicles.append(vehicle)
                            print(f"Created Agent 2 on same lane as Agent 1 but different position")
                        except Exception as e:
                            print(f"Error creating second agent on same lane: {e}")
            
            # Make sure we have at least two vehicles
            while len(self.controlled_vehicles) < 2:
                if self.controlled_vehicles:
                    # Clone the first vehicle with different position
                    vehicle = self.controlled_vehicles[0]
                    try:
                        new_vehicle = vehicle_class(
                            road=road,
                            position=vehicle.position + np.array([20, 0]),
                            heading=vehicle.heading,
                            speed=vehicle.speed
                        )
                        road.vehicles.append(new_vehicle)
                        self.controlled_vehicles.append(new_vehicle)
                        print(f"Created cloned agent")
                    except Exception as e:
                        print(f"Error cloning vehicle: {e}")
                        break
                else:
                    # Create a default vehicle if none exist
                    try:
                        pos = np.array([50, 0])
                        vehicle = vehicle_class(
                            road=road,
                            position=pos,
                            heading=0,
                            speed=20
                        )
                        road.vehicles.append(vehicle)
                        self.controlled_vehicles.append(vehicle)
                        print(f"Created default agent at position {pos}")
                    except Exception as e:
                        print(f"Error creating default vehicle: {e}")
                        break
            
            # Update the environment's controlled vehicles
            if hasattr(self.env.unwrapped, 'controlled_vehicles'):
                self.env.unwrapped.controlled_vehicles = self.controlled_vehicles
            if hasattr(self.env.unwrapped, 'vehicle'):
                self.env.unwrapped.vehicle = self.controlled_vehicles[0]
            
        except Exception as e:
            print(f"Error setting up two-way vehicles: {e}")
            import traceback
            traceback.print_exc()
            
            # Fall back to default vehicles
            if not self.controlled_vehicles and hasattr(self.env.unwrapped, 'vehicle'):
                self.controlled_vehicles = [self.env.unwrapped.vehicle]
            
            while len(self.controlled_vehicles) < self.num_agents:
                if self.controlled_vehicles:
                    # Just duplicate the first vehicle
                    self.controlled_vehicles.append(self.controlled_vehicles[0])
                else:
                    print("Warning: No vehicles available")
                    break
    
    def _create_fallback_vehicle(self, road, source_road, vehicle_class):
        """Create a fallback vehicle if primary placement fails."""
        try:
            # Try to find any valid lane
            for lane_id in range(4):  # Try up to 4 lanes
                lane_index = (source_road[0], source_road[1], lane_id)
                try:
                    lane = road.network.get_lane(lane_index)
                    if lane:
                        # Create vehicle
                        longitudinal_pos = 45  # Different position
                        vehicle_pos = lane.position(longitudinal_pos, 0)
                        heading = lane.heading_at(longitudinal_pos)
                        
                        vehicle = vehicle_class(
                            road=road,
                            position=vehicle_pos,
                            heading=heading,
                            speed=19
                        )
                        
                        # Set speed index
                        if not hasattr(vehicle, 'speed_index'):
                            vehicle.speed_index = 2
                            
                        road.vehicles.append(vehicle)
                        self.controlled_vehicles.append(vehicle)
                        
                        # Set lane
                        vehicle.target_lane_index = lane_index
                        vehicle.lane_index = lane_index
                        print(f"Created fallback agent on lane {lane_id}")
                        return True
                except Exception:
                    continue
        except Exception as e:
            print(f"Error creating fallback vehicle: {e}")
        return False
        
    def reset(self) -> List[np.ndarray]:
        """
        Reset the environment and return initial observations for all agents.
        
        Returns:
            List of initial observations for each agent
        """
        self.reset_count += 1
        print(f"\n--- Environment Reset #{self.reset_count} ---")
        
        # Reset the environment
        obs, info = self.env.reset()
        self.current_step = 0
        self.previous_actions = [0, 0]
        self.action_repeat_count = [0, 0]
        self.allow_early_reset = False  # Reset the early reset flag
        
        # Initialize vehicles in proper positions
        self._initialize_vehicles()
        
        # Get observations for all agents
        observations = []
        for i in range(self.num_agents):
            if i < len(self.controlled_vehicles):
                # Get observation directly from the environment
                if i == 0:
                    observations.append(obs)
                else:
                    # Get a fresh observation for the second agent
                    try:
                        # Switch to this vehicle temporarily
                        original_vehicle = self.env.unwrapped.vehicle
                        self.env.unwrapped.vehicle = self.controlled_vehicles[i]
                        
                        # Get observation for this vehicle
                        if hasattr(self.env.unwrapped, 'observation_type'):
                            agent_obs = self.env.unwrapped.observation_type.observe()
                            observations.append(agent_obs)
                        else:
                            # Fallback to modified first observation
                            modified_obs = obs.copy()
                            observations.append(modified_obs)
                        
                        # Switch back to original vehicle
                        self.env.unwrapped.vehicle = original_vehicle
                    except Exception as e:
                        print(f"Error getting observation for agent {i}: {e}")
                        # Fallback to modified first observation
                        modified_obs = obs.copy()
                        observations.append(modified_obs)
            else:
                # If not enough vehicles, duplicate the first observation
                observations.append(obs)
        
        # Ensure viewer is initialized
        self.render()
        
        # Print agent info
        for i, vehicle in enumerate(self.controlled_vehicles[:self.num_agents]):
            pos = getattr(vehicle, 'position', 'unknown')
            lane = getattr(vehicle, 'lane_index', 'unknown')
            print(f"Agent {i+1} ({self.agent_names[i]}): Position {pos}, Lane {lane}")
        
        return observations
    
    def _get_smart_action(self, agent_idx, proposed_action):
        """Enhanced collision avoidance"""
        # Add safe distance buffer
        SAFE_DISTANCE = 25
        EMERGENCY_BRAKE_DISTANCE = 15
        
        if len(self.controlled_vehicles) > agent_idx:
            vehicle = self.controlled_vehicles[agent_idx]
            
            # Check if vehicle and road are valid
            if vehicle is None or not hasattr(vehicle, 'road') or vehicle.road is None:
                return proposed_action
            
            # Calculate distances to all vehicles
            distances = []
            for other in vehicle.road.vehicles:
                if other is not vehicle and other is not None and not getattr(other, 'crashed', False):
                    try:
                        rel_pos = other.position - vehicle.position
                        distance = np.linalg.norm(rel_pos)
                        angle = np.arctan2(rel_pos[1], rel_pos[0]) - vehicle.heading
                        
                        # Only consider vehicles in front (+/- 45 degree cone)
                        if abs(angle) < np.pi/4 and distance < SAFE_DISTANCE:
                            distances.append(distance)
                    except Exception as e:
                        # Skip this vehicle if any attribute access fails
                        continue

            # Collision avoidance logic
            if distances:
                min_distance = min(distances)
                if min_distance < EMERGENCY_BRAKE_DISTANCE:
                    return 1  # SLOWER
                elif min_distance < SAFE_DISTANCE:
                    # Alternate between slowing and lane change
                    if proposed_action in [3, 4]:  # Lane change actions
                        return proposed_action
                    return 0 if np.random.random() < 0.7 else 1  # Mostly IDLE, sometimes SLOWER

        return proposed_action
    
    def step(self, actions: List[Any]) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool], List[Dict]]:
        """
        Take a step in the environment with actions from all agents.
        
        Args:
            actions: List of actions for each agent
            
        Returns:
            Tuple containing:
            - List of observations for each agent
            - List of rewards for each agent
            - List of done flags for each agent
            - List of truncated flags for each agent
            - List of info dictionaries for each agent
        """
        assert len(actions) == self.num_agents, f"Expected {self.num_agents} actions, got {len(actions)}"
        
        self.current_step += 1
        
        # Make sure we still have our controlled vehicles
        if len(self.controlled_vehicles) < self.num_agents:
            print(f"Warning: Only {len(self.controlled_vehicles)} controlled vehicles available")
            self._update_controlled_vehicles()
            
            # If we still don't have enough vehicles, try to reset
            if len(self.controlled_vehicles) < self.num_agents:
                print("Resetting environment due to missing vehicles")
                return self.reset(), [0.0] * self.num_agents, [True] * self.num_agents, [True] * self.num_agents, [{}] * self.num_agents
        
        # Apply smart action modifications to avoid crashes
        smart_actions = []
        for i, action in enumerate(actions):
            try:
                smart_action = self._get_smart_action(i, action)
                smart_actions.append(smart_action)
            except Exception as e:
                print(f"Error getting smart action for agent {i}: {e}")
                smart_actions.append(action)  # Use original action as fallback
        
        # Take the step with the first vehicle
        try:
            next_obs, reward, done, truncated, info = self.env.step(smart_actions[0])
            
            # If we're below the minimum steps, prevent early termination
            if self.current_step < self.min_steps_before_reset and not self.allow_early_reset:
                # Override done and truncated flags to prevent early termination
                done = False
                truncated = False
                
                # If the vehicle crashed but we want to continue, reset its position
                if hasattr(self.env.unwrapped, 'vehicle') and hasattr(self.env.unwrapped.vehicle, 'crashed') and self.env.unwrapped.vehicle.crashed:
                    # Try to reset the vehicle's position without full environment reset
                    try:
                        # Get the vehicle's current lane
                        lane_index = self.env.unwrapped.vehicle.lane_index
                        lane = self.env.unwrapped.road.network.get_lane(lane_index)
                        
                        # Reset position and clear crashed flag
                        self.env.unwrapped.vehicle.crashed = False
                        self.env.unwrapped.vehicle.position = lane.position(30, 0)  # Reset to start position
                        self.env.unwrapped.vehicle.heading = lane.heading_at(30)
                        self.env.unwrapped.vehicle.speed = 20  # Reset speed
                        
                        # Add negative reward for crash but allow to continue
                        reward = -1.0
                        
                        # Get fresh observation
                        if hasattr(self.env.unwrapped, 'observation_type'):
                            next_obs = self.env.unwrapped.observation_type.observe()
                    except Exception as e:
                        print(f"Error resetting crashed vehicle: {e}")
            elif self.current_step >= self.min_steps_before_reset:
                # After minimum steps, allow resets to happen
                self.allow_early_reset = True
        except Exception as e:
            print(f"Error stepping environment: {e}")
            # Fallback to default observations and rewards
            next_obs = np.zeros(self.observation_space.shape) if hasattr(self.observation_space, 'shape') else np.zeros(10)
            reward = 0.0
            done = True
            truncated = True
            info = {}
        
        # Create result lists
        observations = [next_obs]
        rewards = [reward]
        dones = [done]
        truncateds = [truncated]
        infos = [info]
        
        # Second agent processing
        if len(self.controlled_vehicles) > 1 and len(smart_actions) > 1:
            try:
                # Store original vehicle
                original_vehicle = self.env.unwrapped.vehicle
                
                # Switch to second vehicle
                self.env.unwrapped.vehicle = self.controlled_vehicles[1]
                
                # Apply the action to the second vehicle directly
                if hasattr(self.env.unwrapped.vehicle, 'act'):
                    self.env.unwrapped.vehicle.act(smart_actions[1])
                
                # Get observation and reward for second agent
                second_obs = None
                second_reward = 0.0
                
                try:
                    if hasattr(self.env.unwrapped, 'observation_type'):
                        second_obs = self.env.unwrapped.observation_type.observe()
                        
                        # Dynamic second-agent reward: match environment reward settings
                        env_conf = self.config['environment']
                        collision_reward = env_conf.get('collision_reward', -5.0)
                        high_speed_reward = env_conf.get('high_speed_reward', 0.1)
                        lane_change_penalty = env_conf.get('lane_change_reward', -0.2)
                        vehicle = self.env.unwrapped.vehicle
                        # Collision penalty
                        if hasattr(vehicle, 'crashed') and vehicle.crashed:
                            second_reward = collision_reward
                        else:
                            # High-speed component
                            speed = getattr(vehicle, 'speed', 0.0)
                            max_speed = getattr(vehicle, 'max_speed', 1.0)
                            speed_ratio = speed / max_speed if max_speed > 0 else 0.0
                            second_reward = high_speed_reward * speed_ratio
                            # Lane-change penalty if action was lane change
                            if smart_actions[1] in [3, 4]:
                                second_reward += lane_change_penalty
                except Exception as e:
                    print(f"Error calculating second agent reward: {e}")
                    
                # If observation is still None, use fallback
                if second_obs is None:
                    second_obs = next_obs.copy() + np.random.normal(0, 0.05, next_obs.shape)
                
                # Switch back to original vehicle
                self.env.unwrapped.vehicle = original_vehicle
                
                # Add to result lists
                observations.append(second_obs)
                rewards.append(second_reward)
                dones.append(done)
                truncateds.append(truncated)
                infos.append(info.copy())
            except Exception as e:
                print(f"Error processing second agent: {e}")
                # Fallback to simpler approach
                second_obs = next_obs.copy() + np.random.normal(0, 0.05, next_obs.shape)
                second_reward = reward * 0.95
                
                observations.append(second_obs)
                rewards.append(second_reward)
                dones.append(done)
                truncateds.append(truncated)
                infos.append(info.copy())
        else:
            # Fallback for second agent if not available
            second_obs = next_obs.copy() + np.random.normal(0, 0.05, next_obs.shape)
            second_reward = reward * 0.95
            
            observations.append(second_obs)
            rewards.append(second_reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info.copy())
        
        # End episode after max steps
        if self.current_step >= self.max_steps:
            dones = [True] * self.num_agents
            truncateds = [True] * self.num_agents
            for info in infos:
                info['timeout'] = True
        
        # Apply visual updates to make vehicles distinct
        self._modify_controlled_vehicles()
        
        if any(dones):
            print(f"Episode ending at step {self.current_step}")
            # Reset the early reset flag for next episode
            self.allow_early_reset = False
        
        return observations, rewards, dones, truncateds, infos
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            RGB array or None depending on render mode
        """
        try:
            # Highlight the controlled vehicles for better visualization
            self._modify_controlled_vehicles()
            
            return self.env.render()
        except Exception as e:
            print(f"Warning: Rendering failed: {e}")
            return None
    
    def close(self) -> None:
        """
        Close the environment.
        """
        self.env.close()


if __name__ == "__main__":
    # Simple test of the environment
    env = MultiAgentHighwayEnv()
    observations = env.reset()
    
    print(f"Number of agents: {env.num_agents}")
    
    for step in range(300):
        # Mix of actions for more interesting behavior
        if step % 15 == 0:  # Occasionally change lanes
            actions = [3, 4]  # LANE_LEFT for agent 1, LANE_RIGHT for agent 2
        elif step % 25 == 0:  # Sometimes slow down
            actions = [1, 1]  # SLOWER for both agents
        elif np.random.random() < 0.7:  # Usually go forward
            actions = [2, 2]  # FASTER action for both agents
        else:  # Sometimes random actions
            actions = [env.action_space.sample() for _ in range(env.num_agents)]
        
        observations, rewards, dones, truncateds, infos = env.step(actions)
        
        if step % 10 == 0:
            print(f"Step {step}, Rewards: {rewards}")
            
        env.render()
        time.sleep(0.2)  # Longer delay for smoother visualization
        
        if all(dones):
            print("Episode done, resetting...")
            observations = env.reset()
    
    env.close() 