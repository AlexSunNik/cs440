# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to couter-clockwise

        Return:
            End position (int,int):of the arm link, (x-coordinate, y-coordinate)
    """
    x_offset = int(np.cos(np.radians(angle)) * length)  #use int not round
    y_offset = int(np.sin(np.radians(angle)) * length)
    return (start[0] + x_offset, start[1] - y_offset)

def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all arm links [(start, end, distance)]
            objects (list): x-, y- coordinate and radius of object (obstacles or goals) [(x, y, r)]
            isGoal (bool): True if the object is a goal and False if the object is an obstacle.
                           When the object is an obstacle, consider padding distance.
                           When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """
    #Loop through obj
    for obj in objects:
        x0,y0,r = obj
        #Loop through arm link
        for arm_link in armPosDist:
            start, end, pad_dist = arm_link
            x1,y1 = start
            x2,y2 = end
            dx = x2 - x1
            dy = y2 - y1
            vec_1_to_2 = np.array([dx, dy])
            vec_1_to_0 = np.array([x0 - x1, y0 - y1])
            vec_2_to_0 = np.array([x0 - x2, y0 - y2])
            dot_1_to_0 = np.dot(vec_1_to_2, vec_1_to_0)
            dot_2_to_0 = np.dot(vec_1_to_2 * -1, vec_2_to_0)
            if dot_1_to_0 > 0 and dot_2_to_0 > 0:
                numerator = np.abs(dy*x0 - dx*y0 + x2*y1 - y2*x1)
                denom = np.sqrt(dy**2 + dx**2)
                dist = numerator / denom  # Perpendicular distance from obj to line
            elif dot_1_to_0 <= 0:
                dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            elif dot_2_to_0 <= 0:
                dist = np.sqrt((x2 - x0)**2 + (y2 - y0)**2)
            
            if isGoal:
                if dist <= r:
                    return True
            else:
                if dist <= (r + pad_dist):
                    return True
    return False

def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tip touch goals

        Args:
            armEnd (tuple): the arm tip position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. There can be more than one goal.
        Return:
            True if arm tick touches any goal. False if not.
    """
    for goal in goals:
        dist = np.sqrt((armEnd[0] - goal[0])**2 + (armEnd[1] - goal[1])**2)
        if dist <= goal[2]:
            return True
    return False
    #return True in [((armEnd[0] - goal[0])**2 + (armEnd[1] - goal[1])**2)**0.5 <= goal[2] for goal in goals]


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """
    for arm_pos in armPos:
        start = arm_pos[0]
        end = arm_pos[1]
        if start[0] < 0 or start[0] > window[0] or start[1] < 0 or start[1] > window[1]:
            return False
        if end[0] < 0 or end[0] > window[0] or end[1] < 0 or end[1] > window[1]:
            return False
    return True


if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTipTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")
