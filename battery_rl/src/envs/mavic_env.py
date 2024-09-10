# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference: https://github.com/cyberbotics/webots

import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, JointState
from std_srvs.srv import Empty
from webots_ros.srv import set_float
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import sys
from controller import Supervisor

try:
    import gym
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install gym==0.21"'
    )

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.2

ACCELERATION_LIMIT = np.array([1.0, 1.0, 1.0]) # m/s, rad/s, m/s
ACCELERATION_LIMIT *= TIME_DELTA


K_VERTICAL_THRUST = 68.5    # with this thrust, the drone lifts.
K_VERTICAL_P = 3.0          # P constant of the vertical PID.
K_ROLL_P = 50.0             # P constant of the roll PID.
K_PITCH_P = 30.0            # P constant of the pitch PID.
K_YAW_P = 2.0
K_X_VELOCITY_P = 1
K_Y_VELOCITY_P = 1
K_X_VELOCITY_I = 0.01
K_Y_VELOCITY_I = 0.01
LIFT_HEIGHT = 1


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class MavicEnv(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=250):
        super().__init__()

        # Environment info
        self.robot_name = 'r1'
        self.num_laser_points = 400
        self.invalid_action_clipping = False
        self.energy_reward_coef = 0.0

        self.upper = 5.0
        self.lower = -5.0

        # Open AI Gym generic
        low = np.array([0, -np.pi, -1, -1, -1, -1], dtype=np.float32)
        high = np.array([10, np.pi, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-np.ones(2), np.ones(2))
        self.spec = gym.envs.registration.EnvSpec(id='MavicEnv-v0', max_episode_steps=max_episode_steps)
        self.max_episode_steps = max_episode_steps

        # Open ROS plugins
        port = "11311"
        rospy.init_node("gym_webots", anonymous=True)

        # Environment specific
        # self.__timestep = int(TIME_DELTA*1000) # int(self.getBasicTimeStep())
        self.__timestep = int(self.getBasicTimeStep())
        self.robot = self.getFromDef(f"{self.robot_name}")

        # Sensors
        self.__gps = self.getDevice('gps')
        self.__gyro = self.getDevice('gyro')
        self.__imu = self.getDevice('inertial unit')

        # Initialize motors
        self.motors = {
            'front_left_propeller': self.getDevice('front left propeller'),
            'front_right_propeller': self.getDevice('front right propeller'),
            'rear_left_propeller': self.getDevice('rear left propeller'),
            'rear_right_propeller': self.getDevice('rear right propeller')
        }

        print("env init done!")

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def step(self, action):
        target = False

        # Apply the robot action
        ###############################################################
        # invalid action masking (clipping) with acceleration limit and linear velocity near the goal
        if self.invalid_action_clipping:
            prev_action = self.prev_state[-3:]
            action_diff = action - prev_action
            action_diff = np.clip(action_diff, -ACCELERATION_LIMIT, ACCELERATION_LIMIT)
            action = prev_action + action_diff
        ###############################################################
        for _ in range(25):
            self.apply_action(action)
            super(Supervisor, self).step(self.__timestep)

        # compute distance and angle to the goal
        distance, theta, z_error = self.compute_distance_theta()
        done, collision = self.observe_collision(z_error)

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True
        elif distance > self.upper*1.1:
            collision = True
            done = True

        motor_velocity = []
        for motor in self.motors.values():
            v = motor.getVelocity()
            motor_velocity.append(v)

        robot_state = [distance, theta, z_error, action[0], action[1], action[2]]
        state = np.array(robot_state)
        reward = self.get_reward(target, collision, action, z_error, motor_velocity)
        self.prev_state = np.array(robot_state)
        return state, reward, done, {"target": target}

    def reset(self):
        self.__vertical_ref = LIFT_HEIGHT
        self.__linear_x_integral = 0
        self.__linear_y_integral = 0
        
        # Reset the simulation
        for _ in range(25):
            self.simulationResetPhysics()
            for motor in self.motors.values():
                motor.setPosition(float('inf'))
                motor.setVelocity(0)
            self.__gps.enable(self.__timestep)
            self.__gyro.enable(self.__timestep)
            self.__imu.enable(self.__timestep)
            super(Supervisor, self).step(self.__timestep)

        x, y = 0, 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        new_position = [x, y, 0.1]
        position = self.robot.getField('translation')
        position.setSFVec3f(new_position)

        self.odom_x = x
        self.odom_y = y

        # set a random goal in empty space in environment
        self.change_goal()

        angle = np.random.uniform(-np.pi, np.pi)
        new_rotation = [0, 0, np.sign(angle), np.abs(angle)]
        orientation = self.robot.getField('rotation')
        orientation.setSFRotation(new_rotation)

        # Reset the simulation
        for _ in range(25):
            self.simulationResetPhysics()
            for motor in self.motors.values():
                motor.setPosition(float('inf'))
                motor.setVelocity(0)
            self.__gps.enable(self.__timestep)
            self.__gyro.enable(self.__timestep)
            self.__imu.enable(self.__timestep)
            super(Supervisor, self).step(self.__timestep)

        for _ in range(500):
            self.apply_action([0, 0, 0])
            super(Supervisor, self).step(self.__timestep)

        distance, theta, z_error = self.compute_distance_theta()

        robot_state = [distance, theta, z_error, 0.0, 0.0, 0.0]
        self.prev_state = np.array(robot_state)
        state = np.array(robot_state)
        return state
    
    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            self.goal_z = 1.0
            goal_ok = check_pos(self.goal_x, self.goal_y)

        ##################
        ### Visualize goal on the simulator
        goal = self.getFromDef('CAT')
        position = goal.getField('translation')
        new_position = [self.goal_x, self.goal_y, 0]
        position.setSFVec3f(new_position)
        ##################

    def compute_distance_theta(self):
        position = self.robot.getField('translation').getSFVec3f()
        self.odom_x = position[0]
        self.odom_y = position[1]
        self.odom_z = position[2]

        x_r = self.goal_x - self.odom_x
        y_r = self.goal_y - self.odom_y
        z_error = self.goal_z - self.odom_z
        if x_r == 0 and y_r == 0:
            return 0.0, 0.0
        
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        rotation = self.robot.getField('rotation').getSFRotation()
        angle = rotation[3]*np.sign(rotation[2])

        angle_dif = math.atan2(y_r, x_r)
        theta = angle_dif - angle

        if theta > np.pi:
            theta = theta - (2*np.pi)
        if theta < -np.pi:
            theta = theta + (2*np.pi)

        return distance, theta, z_error
    
    def set_motor_velocity(self, motor_name, velocity):
        rospy.wait_for_service(f'/r1/{motor_name}/set_velocity')
        try:
            set_velocity = rospy.ServiceProxy(f'/r1/{motor_name}/set_velocity', set_float)
            response = set_velocity(velocity)
            return response
        except rospy.ServiceException as e:
            rospy.logerr('Service call failed: %s' % e)
            return None

    def apply_action(self, action):
        roll_ref = 0
        pitch_ref = 0

        # Read sensors
        roll, pitch, _ = self.__imu.getRollPitchYaw()
        _, _, vertical = self.__gps.getValues()
        roll_velocity, pitch_velocity, twist_yaw = self.__gyro.getValues()
        velocity = self.__gps.getSpeed()

        if vertical > 0.2:
            # Calculate velocity
            velocity_x = (pitch / (abs(roll) + abs(pitch))) * velocity
            velocity_y = - (roll / (abs(roll) + abs(pitch))) * velocity

            # High level controller (linear velocity)
            linear_y_error = 0
            linear_x_error = action[0] - velocity_x
            self.__linear_x_integral += linear_x_error
            self.__linear_y_integral += linear_y_error
            roll_ref = K_Y_VELOCITY_P * linear_y_error + K_Y_VELOCITY_I * self.__linear_y_integral
            pitch_ref = - K_X_VELOCITY_P * linear_x_error - K_X_VELOCITY_I * self.__linear_x_integral
            self.__vertical_ref = np.clip(
                self.__vertical_ref + action[2] * (self.__timestep / 1000),
                max(vertical - 0.5, LIFT_HEIGHT),
                vertical + 0.5
            )
        vertical_input = K_VERTICAL_P * (self.__vertical_ref - vertical)

        yaw_ref = action[1]

        roll_input = K_ROLL_P * np.clip(roll, -1, 1) + roll_velocity + roll_ref
        pitch_input = K_PITCH_P * np.clip(pitch, -1, 1) + pitch_velocity + pitch_ref
        yaw_input = K_YAW_P * (yaw_ref - twist_yaw)

        m1 = K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
        m2 = K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
        m3 = K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input
        m4 = K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input

        self.motors['front_right_propeller'].setVelocity(-m1)
        self.motors['front_left_propeller'].setVelocity(m2)
        self.motors['rear_right_propeller'].setVelocity(m3)
        self.motors['rear_left_propeller'].setVelocity(-m4)

    @staticmethod
    def observe_collision(z_error):
        # Detect a collision from laser data
        if np.abs(z_error) > LIFT_HEIGHT / 2:
            return True, True
        return False, False
    
    def get_reward(self, target, collision, action, z_error, motor_velocity):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            motor_velocity_sum = np.sum(np.abs(motor_velocity))
            normalization_factor = 1 / (70*4)
            energy_consumption = motor_velocity_sum * normalization_factor

            return action[0] / 2 - abs(action[1]) / 2 - abs(z_error) / 2 - self.energy_reward_coef*energy_consumption - 0.75