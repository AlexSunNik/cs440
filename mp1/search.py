# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)
import sys

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.
    @param maze: The maze to execute the search on.
    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #Find a single point
    objs = maze.getObjectives()
    start = maze.getStart()
    total_path = []
    for obj in objs:
        queue = []
        visited = {}
        queue.append(start)
        visited[start] = None

        while len(queue) != 0:
            curPt = queue.pop(0)
            if curPt == obj:
                path = []
                while curPt is not None:
                    path.insert(0,curPt)
                    curPt = visited[curPt]
                break
            for p in maze.getNeighbors(curPt[0], curPt[1]): #See neighbors
                if p not in visited and not maze.isWall(curPt[0], curPt[1]):
                    queue.append(p)
                    visited[p] = curPt      #Unlike dijkstra, which considered as visited when expanded
        total_path += path
        start = obj
    return total_path

def Heu_Manhatten(curNode, dest):
    return abs(curNode[0] - dest[0]) + abs(curNode[1] - dest[1])    #Return the manhatten distance

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    dest = maze.getObjectives()[0]
    frontier = {}   #Pair: node : [g_val, f_val]
    prev = {}    #Explored dict
    start = maze.getStart()
    visited = {}  # Explored
    frontier[start] = [0, Heu_Manhatten(start,dest)]
    prev[start] = None
    
    while len(frontier) != 0:
        #Get min f_val pair
        min_pair = min(frontier.items(), key=lambda x: x[1][1]) #Get by f_val
        curNode = min_pair[0]
        del frontier[curNode]
        visited[curNode] = min_pair[1]

        if maze.isObjective(curNode[0], curNode[1]):
            path = []
            while curNode is not None:
                path.insert(0, curNode)
                curNode = prev[curNode]
            return path

        for node in maze.getNeighbors(curNode[0], curNode[1]):
            if maze.isWall(node[0], node[1]):
                continue
            new_g = min_pair[1][0] + 1
            if node in visited and visited[node][0] < new_g:
                continue
            if new_g < frontier.get(node,[sys.maxsize,sys.maxsize])[1]:
                frontier[node] = [new_g, new_g + Heu_Manhatten(node, dest)]
                prev[node] = curNode
    return []

def Heu_Corner(curNode, objs):
    #The function calculates the heuristic function for
    #the four-corners problem
    #It uses the sum of the Manhatten distance
    unvisited = objs[:] #Perform a deep copy
    start = curNode
    heu = 0
    while len(unvisited) != 0:
        nearest_corner = min(unvisited, key=lambda x: Heu_Manhatten(start,x))
        heu += Heu_Manhatten(start, nearest_corner)
        start = nearest_corner
        unvisited.remove(nearest_corner)
    return heu


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.
        
    @param maze: The maze to execute the search on.
        
    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    frontier = {}   #Frontier list
    visited = {}    #Explored list
    total_path = [] #Total path

    start = maze.getStart()
    objs = maze.getObjectives() #Get dests
    prev = {}   #Previous values
    frontier[start] = [0, Heu_Corner(start, objs)]
    path_start = start  #Keep track of the start of the current path
    prev[start] = None

    while len(frontier) != 0:
        #Get min f_val pair
        min_pair = min(frontier.items(), key=lambda x: x[1][1])  # Get by f_val
        curNode = min_pair[0]
        del frontier[curNode]
        visited[curNode] = min_pair[1]

        if curNode in objs:
            path = []   #Current path
            ite = curNode   #Iterator for prev
            while True:
                ite = prev[ite]
                if ite is None:
                    break
                path.insert(0, ite)
            
            total_path += path
            objs.remove(curNode)  # Remove reached objective
            prev = {}
            prev[curNode] = None
            frontier = {}  # Frontier list
            visited = {}  # Explored list
            frontier[curNode] = [0, Heu_Corner(start, objs)]

            if len(objs) == 0:
                total_path.append(curNode)
                return total_path
            
        for node in maze.getNeighbors(curNode[0], curNode[1]):
            if maze.isWall(node[0], node[1]):
                continue
            new_g = min_pair[1][0] + 1
            if node in visited and visited[node][0] < new_g:
                continue
            if new_g < frontier.get(node, [sys.maxsize, sys.maxsize])[1]:  #Tie-breaking here is important
                frontier[node] = [new_g, new_g + Heu_Corner(node, objs)]
                prev[node] = curNode
    return []

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    """
    Initial State:
    """
    # TODO: Write your code here
    return []


def extra(maze):
    """
    Runs extra credit suggestion.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
