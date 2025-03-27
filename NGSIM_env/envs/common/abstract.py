from __future__ import division, print_function, absolute_import
import copy
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from NGSIM_env import utils
from NGSIM_env.envs.common.observation import observation_factory
from NGSIM_env.envs.common.finite_mdp import finite_mdp
from NGSIM_env.envs.common.graphics import EnvViewer
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.control import MDPVehicle
from NGSIM_env.vehicle.dynamics import Obstacle


class AbstractEnv(gym.Env):
    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
    velocity. The action space is fixed, but the observation space and reward function must be defined in the
    environment implementations.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    # A mapping of action indexes to action labels
    ACTIONS = {0: 'LANE_LEFT',
               1: 'IDLE',
               2: 'LANE_RIGHT',
               3: 'FASTER',
               4: 'SLOWER'}

    # A mapping of action labels to action indexes
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    # The maximum distance of any vehicle present in the observation [m]
    PERCEPTION_DISTANCE = 6.0 * MDPVehicle.SPEED_MAX
 

    def __init__(self, config=None, render_mode='rgb_array'):
        # Configuration
        self.config = self.default_config()
        if config:
            self.config.update(config)

        # Seeding
        self.np_random = None
        self.seed()

        # Scene
        self.road = None
        self.vehicle = None

        # Spaces
        self.observation = None
        self.action_space = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self._record_video_wrapper = None
        self.should_update_rendering = True
        self.render_mode = render_mode
        self.offscreen = self.config.get("offscreen_rendering", False)
        self.enable_auto_render = False

        self.reset()

    @classmethod
    def default_config(cls):
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            # "observation": {"type": "TimeToCollision"},
            # "policy_frequency": 1,  # [Hz]
            # "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            # "screen_width": 640,  # [px]
            # "screen_height": 320,  # [px]
            # "centering_position": [0.5, 0.5],
            # "show_trajectories": False,
            
            # TODO: check the observation and action space
            "observation": {"type": "Kinematics"},
            # "action": {"type": "DiscreteMetaAction"},
            "simulation_frequency": 10,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False,
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def configure(self, config):
        if config:
            self.config.update(config)
    
    def update_metadata(self, video_real_time_ratio=2):
        frames_freq = (
            self.config["simulation_frequency"]
            if self._record_video_wrapper
            else self.config["policy_frequency"]
        )
        self.metadata["render_fps"] = video_real_time_ratio * frames_freq

    def define_spaces(self):
        # self.action_space = spaces.Discrete(len(self.ACTIONS))
        # self.action_space = spaces.MultiDiscrete([3, 10])
        self.action_space = spaces.Discrete(30)

        if "observation" not in self.config:
            raise ValueError("The observation configuration must be defined")
        self.observation = observation_factory(self, self.config["observation"])
        self.observation_space = self.observation.space()

    def _reward(self, action):
        """
        Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError
    
    def _rewards(self, action):
        """
        Returns a multi-objective vector of rewards.

        If implemented, this reward vector should be aggregated into a scalar in _reward().
        This vector value should only be returned inside the info dict.

        :param action: the last action performed
        :return: a dict of {'reward_name': reward_value}
        """
        raise NotImplementedError

    def _is_terminal(self):
        """
        Check whether the current state is a terminal state
        :return:is the state terminal
        """
        raise NotImplementedError
    
    def _is_truncated(self):
        """
        Check whether the current state is a truncated state
        :return:is the state truncated
        """
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        """
        Reset the environment to it's initial configuration
        :return: the observation of the reset state
        """
        self.update_metadata()
        self.define_spaces()
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()
        obs = self.observation.observe()
        # info = self._info(obs, action=self.action_space.sample())
        info = None
        if self.render_mode == 'human':
            self.render()

        return obs, info
    
    def _reset(self):
        """
        Reset the state of the environment
        """
        raise NotImplementedError

    def step(self, action):
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default
        behaviour for several simulation timesteps until the next decision making step.
        :param int action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.time += 1 / self.config["policy_frequency"]
        self._simulate(action)

        obs = self.observation.observe()
        reward = self._reward(action)
        terminal = self._is_terminal()
        truncated = self._is_truncated()
        info = self._info(obs, action)
        if self.rendering_mode == 'human':
            self.render()

        return obs, reward, terminal, truncated, info

    def _simulate(self, action=None):
        """
        Perform several steps of simulation with constant action
        """
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])

        for frame in range(frames):
            if action is not None and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                # Forward action to the vehicle
                # TODO: check the action space
                self.vehicle.act(self.ACTIONS[action])

            self.road.act()
            self.road.step(1/self.config["simulation_frequency"])
            self.time += 1

            if frame < frames - 1:
                self._automatic_rendering()

        self.enable_auto_render = False

    def render(self):
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if self.render_mode == 'rgb_array':
            image = self.viewer.get_image()
            return image
                

        self.should_update_rendering = False

    def close(self):
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()

        self.viewer = None

    def get_available_actions(self):
        # TODO: check the available actions
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and velocity changes are not available at
         maximal or minimal velocity.

        :return: the list of available actions
        """
        actions = [self.ACTIONS_INDEXES['IDLE']]
        
        for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
            if l_index[2] < self.vehicle.lane_index[2] and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position):
                actions.append(self.ACTIONS_INDEXES['LANE_LEFT'])
            if l_index[2] > self.vehicle.lane_index[2] and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position):
                actions.append(self.ACTIONS_INDEXES['LANE_RIGHT'])

        if self.vehicle.velocity_index < self.vehicle.SPEED_COUNT - 1:
            actions.append(self.ACTIONS_INDEXES['FASTER'])
        if self.vehicle.velocity_index > 0:
            actions.append(self.ACTIONS_INDEXES['SLOWER'])

        return actions
    
    def set_record_video_wrapper(self, wrapper):
        self._record_video_wrapper = wrapper
        self.update_metadata()
        self._record_video_wrapper.frames_per_sec = self.metadata["render_fps"]

    def _automatic_rendering(self):
        """
        Automatically render the intermediate frames while an action is still ongoing.
        This allows to render the whole video and not only single steps corresponding to agent decision-making.

        If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers
        such as video-recording monitor that need to access these intermediate renderings.
        """
        if self.viewer is not None and self.enable_auto_render:
            if self._record_video_wrapper:
                self._record_video_wrapper._capture_frame()
            else:
                self.render()

    def simplify(self):
        """
        Return a simplified copy of the environment where distant vehicles have been removed from the road.

        This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [
            state_copy.vehicle
        ] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE
        )

        return state_copy

    def change_vehicles(self, vehicle_class_path):
        """
        Change the type of all vehicles on the road
        :param vehicle_class_path: The path of the class of behavior for other vehicles
                             Example: "highway_env.vehicle.behavior.IDMVehicle"
        :return: a new environment with modified behavior model for other vehicles
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle and not isinstance(v, Obstacle):
                vehicles[i] = vehicle_class.create_from(v)
                
        return env_copy

    def set_preferred_lane(self, preferred_lane=None):
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    v.route = [(lane[0], lane[1], preferred_lane) for lane in v.route]
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    def set_route_at_intersection(self, _to):
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)
        return env_copy
    
    def set_vehicle_field(self, args):
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v is not self.vehicle:
                setattr(v, field, value)
        return env_copy

    def call_vehicle_method(self, args):
        method, method_args = args
        env_copy = copy.deepcopy(self)
        for i, v in enumerate(env_copy.road.vehicles):
            if hasattr(v, method):
                env_copy.road.vehicles[i] = getattr(v, method)(*method_args)
        return env_copy

    def randomize_behaviour(self):
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1 / self.config["policy_frequency"])

    def __deepcopy__(self, memo):
        """
        Perform a deep copy but without copying the environment viewer.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)

        return result
