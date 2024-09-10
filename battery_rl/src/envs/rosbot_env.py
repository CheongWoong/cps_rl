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

ACCELERATION_LIMIT = np.array([1.0, 1.0]) # m/s, rad/s
ACCELERATION_LIMIT *= TIME_DELTA


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class RosbotEnv(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=250):
        super().__init__()

        # Environment info
        self.robot_name = 'r1'
        self.num_laser_points = 400
        self.lidar_dim = 20
        self.invalid_action_clipping = False
        self.energy_reward_coef = 0.0

        self.upper = 5.0
        self.lower = -5.0

        # Open AI Gym generic
        low = np.array([0 for _ in range(self.lidar_dim)] + [0, -np.pi, -1, -1], dtype=np.float32)
        high = np.array([10 for _ in range(self.lidar_dim)] + [10, np.pi, 1, 1], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self.action_space = gym.spaces.Box(-np.ones(2), np.ones(2))
        self.spec = gym.envs.registration.EnvSpec(id='RosbotEnv-v0', max_episode_steps=max_episode_steps)
        self.max_episode_steps = max_episode_steps

        # Open ROS plugins
        port = "11311"
        rospy.init_node("gym_webots", anonymous=True)
        subprocess.Popen(["roslaunch", "-p", port, "battery_rl", "prepare_rosbot_plugin.launch", f"robot_name:={self.robot_name}"])
        print("ROSBOT r1 plugin ready!")

        # Set up the ROS publishers and subscribers
        self.laser_scan = rospy.Subscriber(f"{self.robot_name}/laser/laser_scan", LaserScan, self.laser_scan_callback, queue_size=1)
        self.odom = rospy.Subscriber(f"{self.robot_name}/rosbot_diff_drive_controller/odom", Odometry, self.odom_callback, queue_size=1)
        self.joint_states_sub = rospy.Subscriber(f"{self.robot_name}/joint_states", JointState, self.joint_states_callback, queue_size=1)
        self.vel_pub = rospy.Publisher(f"{self.robot_name}/rosbot_diff_drive_controller/cmd_vel", Twist, queue_size=1)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)

        # Environment specific
        self.__timestep = int(TIME_DELTA*1000) # int(self.getBasicTimeStep())
        self.robot = self.getFromDef(f"{self.robot_name}")

        print("env init done!")

    def laser_scan_callback(self, laser_data):
        laser_left = np.array(laser_data.ranges[int(self.num_laser_points*0.75):self.num_laser_points])
        laser_right = np.array(laser_data.ranges[0:int(self.num_laser_points*0.25)])
        laser_state = np.append(laser_left, laser_right)
        laser_state = np.clip(laser_state, 0, 10)
        step_size = len(laser_state) // self.lidar_dim
        laser_state = np.array(
            [min(laser_state[i*step_size:(i+1)*step_size]) for i in range(self.lidar_dim)]
        )
        self.lidar_data = laser_state

    def odom_callback(self, od_data):
        self.last_odom = od_data

    def joint_states_callback(self, data):
        self.joint_states = data.position

    def step(self, action):
        target = False

        # Apply the robot action
        ###############################################################
        # invalid action masking (clipping) with acceleration limit and linear velocity near the goal
        if self.invalid_action_clipping:
            prev_action = self.prev_state[-2:]
            action_diff = action - prev_action
            action_diff = np.clip(action_diff, -ACCELERATION_LIMIT, ACCELERATION_LIMIT)
            action = prev_action + action_diff
        ###############################################################
        self.apply_action(action)
        self.publish_markers(action)

        super(Supervisor, self).step(self.__timestep)

        # read laser state
        done, collision, min_laser = self.observe_collision(self.lidar_data)

        # compute distance and angle to the goal
        distance, theta = self.compute_distance_theta()

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(self.lidar_data, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)
        self.prev_state = np.array(robot_state)
        self.last_joint_states = self.joint_states
        return state, reward, done, {"target": target}

    def reset(self):
        # Reset the simulation
        self.apply_action([0.0, 0.0])
        self.simulationResetPhysics()
        super(Supervisor, self).step(self.__timestep)

        x, y = 0, 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        new_position = [x, y, 0]
        position = self.robot.getField('translation')
        position.setSFVec3f(new_position)

        self.odom_x = x
        self.odom_y = y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0])

        angle = np.random.uniform(-np.pi, np.pi)
        new_rotation = [0, 0, np.sign(angle), np.abs(angle)]
        orientation = self.robot.getField('rotation')
        orientation.setSFRotation(new_rotation)
        
        super(Supervisor, self).step(self.__timestep*4)

        self.last_joint_states = self.joint_states

        distance, theta = self.compute_distance_theta()

        robot_state = [distance, theta, 0.0, 0.0]
        self.prev_state = np.array(robot_state)
        state = np.append(self.lidar_data, robot_state)
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
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(1, 5):
            x, y = 0, 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            
            box = self.getFromDef(f"Box{i}")
            new_position = [x, y, 0]
            position = box.getField('translation')
            position.setSFVec3f(new_position)
            new_rotation = [0, 0, 1, 0]
            orientation = box.getField('rotation')
            # orientation.setSFRotation(new_rotation)

        ##################
        ### Visualize goal on the simulator
        goal = self.getFromDef('CAT')
        position = goal.getField('translation')
        new_position = [self.goal_x, self.goal_y, 0]
        position.setSFVec3f(new_position)
        ##################

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "r1/odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "r1/odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "r1/odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    def compute_distance_theta(self):
        position = self.robot.getField('translation').getSFVec3f()
        self.odom_x = position[0]
        self.odom_y = position[1]

        x_r = self.goal_x - self.odom_x
        y_r = self.goal_y - self.odom_y
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

        return distance, theta
    
    def apply_action(self, action):
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser
    
    def get_reward(self, target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0

            wheel_velocity = (np.array(self.joint_states) - np.array(self.last_joint_states)) / TIME_DELTA
            wheel_velocity_sum = np.sum(np.abs(wheel_velocity))
            MAX_VELOCITY = 26
            normalization_factor = 1 / (MAX_VELOCITY*4)
            energy_consumption = wheel_velocity_sum * normalization_factor

            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2 - self.energy_reward_coef*energy_consumption - 0.75