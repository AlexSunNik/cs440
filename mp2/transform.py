
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

#Author : Xinglong Sun
"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *
"""
def transformToMaze(arm, goals, obstacles, window, granularity):

    #Loop through every angle, and create a map
    limits = arm.getArmLimit()  #(min, max)
    mins = [limit[0] for limit in limits]    #Offsets -- min angles
    maxs = [limit[1] for limit in limits]
    start = arm.getArmAngle()
    #Rows and columns in the maze would match with alpha and beta
    height = round((maxs[0] - mins[0]) / granularity) + 1  # Rows --- Alpha
    width = round((maxs[1] - mins[1]) / granularity) + 1  # Columns --- Beta
    maze_map = np.full(shape = (height, width), fill_value = SPACE_CHAR, dtype = object)
    for alpha in range(mins[0], maxs[0] + granularity, granularity):
        for beta in range(mins[1], maxs[1] + granularity, granularity):
            idxs = angleToIdx([alpha, beta], mins, granularity)
            arm.setArmAngle([alpha, beta])
            pos_dict = arm.getArmPosDist()
            arm_end = arm.getEnd()
            #test obstacles
            if doesArmTouchObjects(pos_dict, obstacles, isGoal=False):
                maze_map[idxs[0], idxs[1]] = WALL_CHAR
                continue
            #test out of bound
            elif not isArmWithinWindow(pos_dict, window):
                maze_map[idxs[0], idxs[1]] = WALL_CHAR
                continue
            #test goal
            elif doesArmTipTouchGoals(arm_end, goals):
                maze_map[idxs[0], idxs[1]] = OBJECTIVE_CHAR
                continue
            #test link hit goals
            elif doesArmTouchObjects(pos_dict, goals, isGoal=True):
                maze_map[idxs[0], idxs[1]] = WALL_CHAR
                continue
            #test space
            else:
                maze_map[idxs[0], idxs[1]] = SPACE_CHAR
                continue
    start_idxs = angleToIdx(start, mins, granularity)
    maze_map[start_idxs[0], start_idxs[1]] = START_CHAR
    arm.setArmAngle([start[0], start[1]])
    maze = Maze(maze_map, mins, granularity)
    return maze
    """

#For extra

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.

        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    #Loop through every angle, and create a map
    limits = arm.getArmLimit()  #(min, max)
    #Append limits
    while len(limits) < 3:
        limits.append((0, 0))

    mins = [limit[0] for limit in limits]    #Offsets -- min angles
    maxs = [limit[1] for limit in limits]
    
    start = list(arm.getArmAngle())
    print(start)
    #Append start
    while len(start) < 3:
        start.append(0)

    #Rows and columns in the maze would match with alpha and beta
    x_len = round((maxs[0] - mins[0]) / granularity) + 1  # Rows --- Alpha
    y_len = round((maxs[1] - mins[1]) / granularity) + 1  # Columns --- Beta
    z_len = round((maxs[2] - mins[2]) / granularity) + 1  # Gamma

    maze_map = np.full(shape=(x_len, y_len, z_len), fill_value=SPACE_CHAR, dtype=object)
    #print(maze_map.shape)
    for alpha in range(mins[0], maxs[0] + granularity, granularity):
        for beta in range(mins[1], maxs[1] + granularity, granularity):
            for gamma in range(mins[2], maxs[2]  + granularity, granularity):
                idxs = angleToIdx([alpha, beta, gamma], mins, granularity)
                arm.setArmAngle([alpha, beta, gamma])
                pos_dict = arm.getArmPosDist()
                arm_end = arm.getEnd()
                #test obstacles
                if doesArmTouchObjects(pos_dict, obstacles, isGoal=False):
                    maze_map[idxs[0], idxs[1], idxs[2]] = WALL_CHAR
                    continue
                #test out of bound
                elif not isArmWithinWindow(pos_dict, window):
                    maze_map[idxs[0], idxs[1], idxs[2]] = WALL_CHAR
                    continue
                #test goal
                elif doesArmTipTouchGoals(arm_end, goals):
                    maze_map[idxs[0], idxs[1], idxs[2]] = OBJECTIVE_CHAR
                    continue
                #test link hit goals
                elif doesArmTouchObjects(pos_dict, goals, isGoal=True):
                    maze_map[idxs[0], idxs[1], idxs[2]] = WALL_CHAR
                    continue
                #test space
                else:
                    maze_map[idxs[0], idxs[1], idxs[2]] = SPACE_CHAR
                    continue
    start_idxs = angleToIdx(start, mins, granularity)
    maze_map[start_idxs[0], start_idxs[1], start_idxs[2]] = START_CHAR
    arm.setArmAngle([start[0], start[1], start[2]])
    maze = Maze(maze_map, mins, granularity)
    print("Create Maze success")
    return maze
