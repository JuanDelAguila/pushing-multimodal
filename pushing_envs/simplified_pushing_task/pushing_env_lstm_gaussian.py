import gym
from gym.spaces import Box
import numpy as np
from collections import deque
import pybullet as pb

class PushingEnv(gym.Env):

    def __init__(self, graphics = False, seed = None, fps = 30):
        super(PushingEnv, self).__init__()

        # Number of previous observations to use in the policy
        self.STACK_SIZE = 1

        # Action space: x velocity, y velocity
        self.action_space = Box(low = -1, high = 1, shape = (2,), dtype = np.float32)
        
        # Observation space: (target_x_box, target_y_box, target_theta), (x_box, y_box, orientation_box) * STACK_SIZE, (x_pusher, y_pusher) * STACK_SIZE
        self.observation_space = Box(low = -1, high = 1, shape = (3 + 3*self.STACK_SIZE + 2*self.STACK_SIZE,), dtype = np.float64)

        # Number of steps ellapsed in the current episode
        self.step_num = 0

        # Store the seed for the random number generator 
        self.seed = seed

        # Determines whether to generate a GUI
        self.graphics = graphics

        # Whether to apply random disturbances during training
        self.disturbances = False


        # Distance between pusher and ground
        self.PUSHER_GROUND_CLEARENCE = 0.01
        # Width of the table (in the x direction)
        self.TABLE_DIM_X = 0.60
        # Depth of the table (in the y direction)
        self.TABLE_DIM_Y = 0.35
        # Height of the box (in the z direction)
        self.BOX_DIM_Z = 0.07

        # Defines an inner box within which the random starting positions and target positions are generated
        self.GEN_POS_MARGIN = 0.1
        # Width of x interval within which start and target positions are generated
        self.GEN_POS_X_WIDTH = self.TABLE_DIM_X/2 - self.GEN_POS_MARGIN
        # Width of y interval within which start and target positions are generated
        self.GEN_POS_Y_WIDTH = self.TABLE_DIM_Y - 2*self.GEN_POS_MARGIN
        # Starting space between the pusher and the box
        self.SPACE_PUSHER_BOX = 1e-3

        # Current step of the curriculum
        self.CURRICULUM_STEP = 1

        # Width of possible starting orientations
        self.GEN_THETA_WIDTH = min(2*np.pi, self.CURRICULUM_STEP*np.pi/2)

        # Distance to target position that is considered successful
        self.DISTANCE_SUCCESS = 0.015
        # Distance to target theta that is considered successful
        self.THETA_DISTANCE_SUCCESS = 0.34

        # Success rate which prompts an increase in difficulty
        self.CURRICULUM_SUCCESS_THRESH = 90
        
        # Number of episodes to calculate current success rate
        self.NUM_EPISODES_SUCCESS_RATE = 100

        # Array to keep track of previous 100 episodes outcome
        self.EPISODE_SUCCESS_RATE = np.zeros(self.NUM_EPISODES_SUCCESS_RATE)

        # Number of episodes completed
        self.EPISODE_NUM = 0

        # Current success rate calculated over the past 100 episodes
        self.CURRENT_SUCCESS_RATE = 0

        # Box that is being pushed
        self.box = None
        # End effector that pushes the box
        self.pusher = None
        # Surface on which the box slides
        self.floor = None
        # Visualization of target
        self.target_vis = None

        # x coordinate of the box at the starting pose
        self.start_x_box = None
        # y coordinate of the box at the starting pose
        self.start_y_box = None
        # Orientation of the box at the starting pose
        self.start_theta = None
        # x coordinate of the box at the target pose
        self.target_x_box = None
        # y coordinate of the box at the target pose
        self.target_y_box = None
        # Orientation of the box at the target pose
        self.target_theta = None

        # Dictionary to store the information about box and pusher in the simulation
        self.sim_data = {
            "x_box" : None,
            "y_box" : None,
            "theta_box" : None,
            "velocity_box" : None,
            "distance_to_target" : None,
            "theta_distance_to_target" : None,
            "x_pusher" : None,
            "y_pusher" : None,
            "velocity_pusher" : None
        }

        # Parameters for domain randomization
        self.FRICTION_CENTER = 0.6
        self.FRICTION_WIDTH = 0.2

        self.RESTITUTION_CENTER = 0.5
        self.RESTITUTION_WIDTH = 0.2

        self.BOX_Y_CENTER = 0.12
        self.BOX_Y_WIDTH = 0.01

        self.BOX_X_CENTER = 0.1
        self.BOX_X_WIDTH = 0.01

        self.BOX_MASS_CENTER = 0.5
        self.BOX_MASS_WIDTH = 0.2

        self.PUSHER_RADIUS_CENTER = 0.0125
        self.PUSHER_RADIUS_WIDTH = 0.001

        self.FORCE_WIDTH = 50

        # Parameters for observation noise
        self.DISTANCE_NOISE_MEAN = 0
        self.DISTANCE_NOISE_STD = 0.001

        self.THETA_NOISE_MEAN = 0
        self.THETA_NOISE_STD = 0.02

        # Episode and step noise quantities
        self.BOX_X_NOISE_EPISODE = None
        self.BOX_X_NOISE_STEP = None
        self.BOX_Y_NOISE_EPISODE = None
        self.BOX_Y_NOISE_STEP = None
        self.BOX_THETA_NOISE_EPISODE = None
        self.BOX_THETA_NOISE_STEP = None
        self.PUSHER_X_NOISE_EPISODE = None
        self.PUSHER_X_NOISE_STEP = None
        self.PUSHER_Y_NOISE_EPISODE = None
        self.PUSHER_Y_NOISE_STEP = None

        # Stack of previous box poses
        self.box_pose_stack = None

        # Stack of previous pusher positions
        self.pusher_pos_stack = None

        # Number of pybullet steps for every agent step
        self.pybullet_steps = int(240 / fps)

        # Maximum number of steps before the environment is reset
        self.max_episode_length = 300

        # Setup PyBullet
        if self.graphics:
            pb.connect(pb.GUI) # Generate a graphical interface
            self._draw_boundary() # Draw the table boundaries
        else:
            pb.connect(pb.DIRECT) # Communicate directly with the physics engine
        pb.setGravity(0,0,-9.81)

        # Reset the environment
        self.reset()

    def step(self, action):

        # Scale the action to the desired range
        action_scale = self._adjust_action(action)
        self.velocity_x = action_scale[0]
        self.velocity_y = action_scale[1]

        # Create a random disturbance to the box
        if self.disturbances and (self.rng.random() < 0.01) and self._can_apply_random_disturbance():
            self._apply_random_disturbance()

        # Update simulation
        pb.resetBaseVelocity(objectUniqueId = self.pusher, linearVelocity = [self.velocity_x, self.velocity_y, 0])
        action_steps = int(np.around(self.rng.normal(self.pybullet_steps,0.75), decimals = 0))
        if action_steps < 0:
            action_steps = self.pybullet_steps
        for _ in range(action_steps):
            pb.stepSimulation()
        self._update_sim_data()

        # Calculate reward
        reward, done = self._calculate_reward()

        # Obtain new observation
        observation = self._get_observation()

        # Log relevant information
        info = {}

        # Timeout requires careful consideration.
        self.step_num += 1
        if (not done) and (self.step_num >= self.max_episode_length):
            info["TimeLimit.truncated"] = True
            done = True
            self._record_unsuccessful_episode()

        return observation, reward, done, info

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

        self.step_num = 0

        # Remove the box, pusher, target and floor from the simulator
        if self.box != None:
            pb.removeBody(self.box)
        if self.pusher != None:
            pb.removeBody(self.pusher)
        if self.target_vis != None:
            pb.removeBody(self.target_vis)
        if self.floor != None:
            self._reset_floor()
        else:
            self._create_floor()

        # Generate random dimensions of the box
        self.BOX_DIM_X = self.BOX_X_CENTER + self.rng.random()*self.BOX_X_WIDTH - self.BOX_X_WIDTH/2
        self.BOX_DIM_Y = self.BOX_Y_CENTER + self.rng.random()*self.BOX_Y_WIDTH - self.BOX_Y_WIDTH/2
        self.BOX_RADIUS = self.BOX_DIM_X/2 # Inner radius (used to check if the box is inside the boundary)

        # Generate random dimensions of the pusher
        self.PUSHER_RADIUS = self.PUSHER_RADIUS_CENTER + self.rng.random()*self.PUSHER_RADIUS_WIDTH - self.PUSHER_RADIUS_WIDTH/2

        # Generate episode noise for the observations
        self.BOX_X_NOISE_EPISODE = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.BOX_Y_NOISE_EPISODE = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.BOX_THETA_NOISE_EPISODE = self.rng.normal(self.THETA_NOISE_MEAN, self.THETA_NOISE_STD)
        self.PUSHER_X_NOISE_EPISODE = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.PUSHER_Y_NOISE_EPISODE = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)

        # Generate a random starting configuration
        self.start_x_box, self.start_y_box, self.start_theta, pusher_x, pusher_y = self._generate_random_start()
        self._create_box(x_start=self.start_x_box, y_start=self.start_y_box, theta_start=self.start_theta)
        self._create_pusher(x_start=pusher_x, y_start=pusher_y)
        self.box_pose_stack = deque([[self.start_x_box, self.start_y_box, self.start_theta] for _ in range(self.STACK_SIZE)], maxlen=self.STACK_SIZE)
        self.pusher_pos_stack = deque([[pusher_x, pusher_y] for _ in range(self.STACK_SIZE)], maxlen=self.STACK_SIZE)

        # Generate a random target
        self.target_x_box, self.target_y_box, self.target_theta = self._generate_random_target()

        # Normalize target
        target_x_box_norm = self.target_x_box / (self.TABLE_DIM_X/2)
        target_y_box_norm = self.target_y_box / (self.TABLE_DIM_Y/2)
        target_theta_norm = self.target_theta / np.pi

        # Target observation
        self.target_observation = np.array([target_x_box_norm, target_y_box_norm, target_theta_norm])

        # Update information of simulation objects
        self._update_sim_data()     
        
        # Obtain observation of the environment
        observation = self._get_observation()

        return observation

    def curriculum_take_step (self):
        self.CURRICULUM_STEP += 1
        self.GEN_THETA_WIDTH = min(2*np.pi, self.GEN_THETA_WIDTH + np.pi/2)


    def _update_sim_data(self):
        # Query PyBullet for box data
        box_position_and_orientation = pb.getBasePositionAndOrientation(self.box)
        x_box, y_box, _ = box_position_and_orientation[0]
        theta_box = pb.getEulerFromQuaternion(box_position_and_orientation[1])[2]
        velociy_box, _ = pb.getBaseVelocity(self.box)
        velociy_box = np.linalg.norm(velociy_box[:2])

        distance_to_target = self._distance_to(x_box, y_box, self.target_x_box, self.target_y_box)

        theta_distance_to_target = self._theta_distance(self.target_theta, theta_box)

        # Query PyBullet for pusher data
        pusher_position_orientation = pb.getBasePositionAndOrientation(self.pusher)
        x_pusher, y_pusher, _ = pusher_position_orientation[0]
        velociy_pusher, _ = pb.getBaseVelocity(self.pusher)
        velociy_pusher = np.linalg.norm(velociy_pusher[:2])

        # Update records of simulation data
        self.sim_data["x_box"] = x_box
        self.sim_data["y_box"] = y_box
        self.sim_data["theta_box"] = theta_box
        self.sim_data["velocity_box"] = velociy_box
        self.sim_data["distance_to_target"] = distance_to_target
        self.sim_data["theta_distance_to_target"] = theta_distance_to_target
        self.sim_data["x_pusher"] = x_pusher
        self.sim_data["y_pusher"] = y_pusher
        self.sim_data["velocity_pusher"] = velociy_pusher
        

    def _can_apply_random_disturbance(self):
        if self.sim_data["distance_to_target"] < 0.15:
            return False
        if self.TABLE_DIM_X/2 - abs(self.sim_data["x_box"]) < 0.1:
            return False
        if self.TABLE_DIM_Y/2 - abs(self.sim_data["y_box"]) < 0.1:
            return False
        return True

    def _apply_random_disturbance(self):

        # Generate random force direction
        force_x = self.rng.random()*self.FORCE_WIDTH - self.FORCE_WIDTH/2
        force_y = self.rng.random()*self.FORCE_WIDTH - self.FORCE_WIDTH/2

        # Generate random position at which to apply the force
        pos_x = self.rng.random()*self.BOX_DIM_X - self.BOX_DIM_X/2
        pos_y = self.rng.random()*self.BOX_DIM_Y - self.BOX_DIM_Y/2
        pos_z = self.rng.random()*self.BOX_DIM_Z - self.BOX_DIM_Z/2

        # Apply the force (direction and position relative to the box)
        pb.applyExternalForce(objectUniqueId = self.box, 
                              linkIndex = -1,
                              forceObj = [force_x, force_y, 0],
                              posObj = [pos_x, pos_y, pos_z], 
                              flags = pb.LINK_FRAME)

    def _record_unsuccessful_episode(self):
        self.EPISODE_SUCCESS_RATE[self.EPISODE_NUM%self.NUM_EPISODES_SUCCESS_RATE] = 0
        self.EPISODE_NUM += 1
        self.CURRENT_SUCCESS_RATE = 100*sum(self.EPISODE_SUCCESS_RATE) / self.NUM_EPISODES_SUCCESS_RATE

    def _record_successful_episode(self):
        self.EPISODE_SUCCESS_RATE[self.EPISODE_NUM%self.NUM_EPISODES_SUCCESS_RATE] = 1
        self.EPISODE_NUM += 1
        self.CURRENT_SUCCESS_RATE = 100*sum(self.EPISODE_SUCCESS_RATE) / self.NUM_EPISODES_SUCCESS_RATE

    def _distance_to(self, x1, y1, x2, y2):
        return ((x2-x1)**2 + (y2-y1)**2)**(1/2)

    def _theta_distance(self, theta1, theta2):
        theta_distance = abs(theta1-theta2)
        if theta_distance <= np.pi:
            return theta_distance
        else:
            return (2*np.pi - theta_distance)

    def _create_floor(self):
        floor_visual = pb.createVisualShape(shapeType = pb.GEOM_PLANE, planeNormal = [0, 0, 1])
        floor_collision = pb.createCollisionShape(shapeType = pb.GEOM_PLANE, planeNormal = [0, 0, 1])
        self.floor = pb.createMultiBody(
            baseMass = 0,
            baseCollisionShapeIndex = floor_collision,
            baseVisualShapeIndex = floor_visual
        )
        pb.changeDynamics(
            bodyUniqueId = self.floor,
            linkIndex = -1,
            lateralFriction = self.FRICTION_CENTER + self.rng.random()*self.FRICTION_WIDTH - self.FRICTION_WIDTH/2,
            spinningFriction = 0,
            rollingFriction = 0,
            restitution = self.RESTITUTION_CENTER + self.rng.random()*self.RESTITUTION_WIDTH - self.RESTITUTION_WIDTH/2
        )
    
    def _reset_floor(self):
        pb.changeDynamics(
            bodyUniqueId = self.floor,
            linkIndex = -1,
            lateralFriction = self.FRICTION_CENTER + self.rng.random()*self.FRICTION_WIDTH - self.FRICTION_WIDTH/2,
            spinningFriction = 0,
            rollingFriction = 0,
            restitution = self.RESTITUTION_CENTER + self.rng.random()*self.RESTITUTION_WIDTH - self.RESTITUTION_WIDTH/2
        )

    def _create_box(self, x_start, y_start, theta_start):
        box_visual = pb.createVisualShape(shapeType = pb.GEOM_BOX, halfExtents = [self.BOX_DIM_X/2, self.BOX_DIM_Y/2, self.BOX_DIM_Z/2], rgbaColor = [1, 1, 0, 1])
        box_collision = pb.createCollisionShape(shapeType = pb.GEOM_BOX, halfExtents = [self.BOX_DIM_X/2, self.BOX_DIM_Y/2, self.BOX_DIM_Z/2])
        self.box = pb.createMultiBody(
            baseMass = self.BOX_MASS_CENTER + self.rng.random()*self.BOX_MASS_WIDTH - self.BOX_MASS_WIDTH/2,
            baseCollisionShapeIndex = box_collision,
            baseVisualShapeIndex = box_visual,
            basePosition = [x_start, y_start, self.BOX_DIM_Z/2],
            baseOrientation = pb.getQuaternionFromEuler([0, 0, theta_start])
        )
        pb.changeDynamics(
            bodyUniqueId = self.box,
            linkIndex = -1,
            lateralFriction = self.FRICTION_CENTER + self.rng.random()*self.FRICTION_WIDTH - self.FRICTION_WIDTH/2,
            spinningFriction = 0,
            rollingFriction = 0,
            restitution = self.RESTITUTION_CENTER + self.rng.random()*self.RESTITUTION_WIDTH - self.RESTITUTION_WIDTH/2
        )

    def _create_pusher(self, x_start, y_start):
        pusher_visual = pb.createVisualShape(shapeType = pb.GEOM_SPHERE, radius = self.PUSHER_RADIUS, rgbaColor = [0, 0, 1, 1])
        pusher_collision = pb.createCollisionShape(shapeType = pb.GEOM_SPHERE, radius = self.PUSHER_RADIUS)
        self.pusher = pb.createMultiBody(
            baseMass = 0,
            baseCollisionShapeIndex = pusher_collision,
            baseVisualShapeIndex = pusher_visual,
            basePosition = [x_start, y_start, self.BOX_DIM_Z/2],
            baseOrientation = pb.getQuaternionFromEuler([0, 0, 0])
        )
        pb.changeDynamics(
            bodyUniqueId = self.pusher,
            linkIndex = -1,
            lateralFriction = self.FRICTION_CENTER + self.rng.random()*self.FRICTION_WIDTH - self.FRICTION_WIDTH/2,
            spinningFriction = 0,
            rollingFriction = 0,
            restitution = self.RESTITUTION_CENTER + self.rng.random()*self.RESTITUTION_WIDTH - self.RESTITUTION_WIDTH/2
        )

    def _draw_boundary(self):
        pb.addUserDebugLine(lineFromXYZ = [-self.TABLE_DIM_X/2, -self.TABLE_DIM_Y/2, 0.01], lineToXYZ = [self.TABLE_DIM_X/2, -self.TABLE_DIM_Y/2, 0.01], lineColorRGB = [0, 0, 0], lineWidth = 5)
        pb.addUserDebugLine(lineFromXYZ = [-self.TABLE_DIM_X/2, self.TABLE_DIM_Y/2, 0.01], lineToXYZ = [self.TABLE_DIM_X/2, self.TABLE_DIM_Y/2, 0.01], lineColorRGB = [0, 0, 0], lineWidth = 5)
        pb.addUserDebugLine(lineFromXYZ = [-self.TABLE_DIM_X/2, -self.TABLE_DIM_Y/2, 0.01], lineToXYZ = [-self.TABLE_DIM_X/2, self.TABLE_DIM_Y/2, 0.01], lineColorRGB = [0, 0, 0], lineWidth = 5)
        pb.addUserDebugLine(lineFromXYZ = [self.TABLE_DIM_X/2, -self.TABLE_DIM_Y/2, 0.01], lineToXYZ = [self.TABLE_DIM_X/2, self.TABLE_DIM_Y/2, 0.01], lineColorRGB = [0, 0, 0], lineWidth = 5)

    def _is_box_in_boundary(self):
        # Checks if the box is within the table boundary.
        x_pos = self.sim_data["x_box"]
        y_pos = self.sim_data["y_box"]

        return (x_pos >= -self.TABLE_DIM_X/2 + self.BOX_RADIUS) and (x_pos <= self.TABLE_DIM_X/2 - self.BOX_RADIUS) and (y_pos >= -self.TABLE_DIM_Y/2 + self.BOX_RADIUS) and (y_pos <= self.TABLE_DIM_Y/2 - self.BOX_RADIUS)

    def _is_pusher_in_boundary(self):
        # Checks if the pusher is within the table boundary.
        x_pos = self.sim_data["x_pusher"]
        y_pos = self.sim_data["y_pusher"]

        return (x_pos >= -self.TABLE_DIM_X/2 + self.PUSHER_RADIUS) and (x_pos <= self.TABLE_DIM_X/2 - self.PUSHER_RADIUS) and (y_pos >= -self.TABLE_DIM_Y/2 + self.PUSHER_RADIUS) and (y_pos <= self.TABLE_DIM_Y/2 - self.PUSHER_RADIUS)

    def _generate_random_start(self):

        # Generate random starting position of the box (box_x, box_y)
        box_x = self.rng.random()*self.GEN_POS_X_WIDTH - self.GEN_POS_X_WIDTH
        box_y = self.rng.random()*self.GEN_POS_Y_WIDTH - self.GEN_POS_Y_WIDTH/2

        # Generate random starting orientation of the box (theta)
        theta = self.rng.random()*self.GEN_THETA_WIDTH - self.GEN_THETA_WIDTH / 2

        # Generate random starting position of the pusher (pusher_x, pusher_y) at the back side of the box
        pusher_start_x = -self.BOX_DIM_X/2 - self.PUSHER_RADIUS - self.SPACE_PUSHER_BOX
        pusher_start_y = self.rng.random()*self.BOX_DIM_Y - self.BOX_DIM_Y/2
        pusher_start = np.array([pusher_start_x, pusher_start_y])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        pusher_rotated = rotation_matrix@pusher_start
        pusher_x, pusher_y = pusher_rotated + np.array([box_x, box_y])

        return box_x, box_y, theta, pusher_x, pusher_y

    def _generate_random_target(self):
        target_x = self.rng.random()*self.GEN_POS_X_WIDTH
        target_y = self.rng.random()*self.GEN_POS_Y_WIDTH - self.GEN_POS_Y_WIDTH/2
        target_theta = self.rng.random()*self.GEN_THETA_WIDTH - self.GEN_THETA_WIDTH / 2

        # Visualize target
        if self.graphics:
            target_visual = pb.createVisualShape(shapeType = pb.GEOM_BOX, halfExtents = [self.BOX_DIM_X/2, self.BOX_DIM_Y/2, self.BOX_DIM_Z/2], rgbaColor = [0, 1, 0, 0.5])
            self.target_vis = pb.createMultiBody(
                baseMass = 0,
                baseVisualShapeIndex = target_visual,
                basePosition = [target_x, target_y, self.BOX_DIM_Z/2],
                baseOrientation = pb.getQuaternionFromEuler([0, 0, target_theta])
            )

        return target_x, target_y, target_theta

    def _calculate_reward(self):

        # value 0 when as far as possible within the table, value 0.1 when at the target
        distance_reward = 0.1*(1 - self.sim_data["distance_to_target"] / (self.TABLE_DIM_X**2 + self.TABLE_DIM_Y**2)**(1/2))

        # value 0 when pointing away from the target, value 0.02 when pointing towards the target
        theta_reward = 0.02*(1 - self.sim_data["theta_distance_to_target"] / np.pi)

        # value 0 when at maximum velocity, value 0.004 when not moving
        velocity_reward = 0.004*(1 - 5*np.sqrt(2)*self.sim_data["velocity_pusher"] + 1e-6)

        # Give a reward in terms of the current distance, orientation, and pusher velocity
        reward = distance_reward + theta_reward + velocity_reward

        # Punish if the box or the pusher leave the table
        if (not self._is_box_in_boundary()) or (not self._is_pusher_in_boundary()):
            self._record_unsuccessful_episode()
            return -20, True
        
        # Reward if the agent is successful
        if (self.sim_data["velocity_box"] <= 1e-4) and (self.sim_data["distance_to_target"] <= self.DISTANCE_SUCCESS) and (self.sim_data["theta_distance_to_target"] <= self.THETA_DISTANCE_SUCCESS):
            self._record_successful_episode()
            return 50, True
        
        return reward, False

    def _get_observation(self):
        # Generate observation noise for the current step
        self.BOX_X_NOISE_STEP = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.BOX_Y_NOISE_STEP = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.BOX_THETA_NOISE_STEP = self.rng.normal(self.THETA_NOISE_MEAN, self.THETA_NOISE_STD)
        self.PUSHER_X_NOISE_STEP = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)
        self.PUSHER_Y_NOISE_STEP = self.rng.normal(self.DISTANCE_NOISE_MEAN, self.DISTANCE_NOISE_STD)

        # Update box pose stack
        self.box_pose_stack.append([(self.sim_data["x_box"] + self.BOX_X_NOISE_EPISODE + self.BOX_X_NOISE_STEP) / (self.TABLE_DIM_X/2), 
                                    (self.sim_data["y_box"] + self.BOX_Y_NOISE_EPISODE + self.BOX_Y_NOISE_STEP) / (self.TABLE_DIM_Y/2),
                                    (self.sim_data["theta_box"] + self.BOX_THETA_NOISE_EPISODE + self.BOX_THETA_NOISE_STEP) / np.pi])

        # Update pusher position stack
        self.pusher_pos_stack.append([(self.sim_data["x_pusher"] + self.PUSHER_X_NOISE_EPISODE + self.PUSHER_X_NOISE_STEP) / (self.TABLE_DIM_X/2),
                                      (self.sim_data["y_pusher"] + self.PUSHER_Y_NOISE_EPISODE + self.PUSHER_Y_NOISE_STEP) / (self.TABLE_DIM_Y/2)])

        return np.concatenate((self.target_observation, np.array(self.box_pose_stack).flatten(), np.array(self.pusher_pos_stack).flatten()))

    def _adjust_action(self, action):
        # We want max velocity on each axis of 0.1.
        return action * 0.1

    def render(self, mode='human'):
        pass
    
    def close (self):
        pb.disconnect()
