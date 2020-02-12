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
from queue import PriorityQueue
import heapq
from copy import deepcopy


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
                    path.insert(0, curPt)
                    curPt = visited[curPt]
                break
            for p in maze.getNeighbors(curPt[0], curPt[1]):  # See neighbors
                if p not in visited and not maze.isWall(curPt[0], curPt[1]):
                    queue.append(p)
                    # Unlike dijkstra, which considered as visited when expanded
                    visited[p] = curPt
        total_path += path
        start = obj
    return total_path


def Heu_Manhatten(curNode, dest):
    # Return the manhatten distance
    return abs(curNode[0] - dest[0]) + abs(curNode[1] - dest[1])


def astar(maze):
    """
    Runs A star for part 1 of the assignment.
    @param maze: The maze to execute the search on.
    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    dest = maze.getObjectives()[0]
    frontier = {}  # Pair: node : [g_val, f_val]
    prev = {}  # Explored dict
    start = maze.getStart()
    visited = {}  # Explored
    frontier[start] = [0, Heu_Manhatten(start, dest)]
    prev[start] = None

    while len(frontier) != 0:
        #Get min f_val pair
        min_pair = min(frontier.items(), key=lambda x: x[1][1])  # Get by f_val
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
            new_g = min_pair[1][0] + 1
            if node in visited:
                continue
            if new_g < frontier.get(node, [sys.maxsize, sys.maxsize])[1]:
                frontier[node] = [new_g, new_g + Heu_Manhatten(node, dest)]
                prev[node] = curNode
    return []


def Heu_Corner(curNode, objs):
    #The function calculates the heuristic function for
    #the four-corners problem
    #It uses the sum of the Manhatten distance
    if len(objs) == 0:
        return 0
    unvisited = objs[:]  # Perform a deep copy
    start = curNode
    heu = 0
    while len(unvisited) != 0:
        nearest_corner = min(unvisited, key=lambda x: Heu_Manhatten(start, x))
        heu += Heu_Manhatten(start, nearest_corner)
        start = nearest_corner
        unvisited.remove(nearest_corner)
    return heu


class state:
    def __init__(self, curNode, g, objs):
        self.node = curNode
        self.g = g
        self.objs = objs[:]  # A deep copy
        #self.prev = None

    def __eq__(self, other):
        if self.node == other.node and set(self.objs) == set(other.objs):
            return True
        else:
            return False

    def __lt__(self, other):
        return self.g < other.g

    def __hash__(self):
        return hash(repr(self))

    def isGoal(self):
        return len(self.objs) == 0

    def getH_corner(self, heuristic):
        return heuristic(self.node, self.objs)

    def getH_multi(self, MST_table, heuristic_table):
        return Find_Mst(self.objs, MST_table, heuristic_table)

    def print_state(self):
        print("Node:", self.node, "Obj list:", self.objs)


def astar_corner(maze):
    #state representation in the class "state"
    frontier = []
    visited = {}  # Mapping from state to g
    #visited = []
    prev = {}  # State to state, keep track of the previous state

    objs = maze.getObjectives()
    start = maze.getStart()

    #Push the starting state
    initState = state(start, 0, objs)
    f_init = Heu_Corner(start, objs)
    heapq.heappush(frontier, (f_init, initState))
    prev[initState] = None

    while True:
        curState = heapq.heappop(frontier)[1]
        #curState.print_state()
        if(curState.isGoal()):
            #Reach the goal state
            #Backtrace to get the path
            path = []
            ite = curState
            while ite is not None:
                path.insert(0, ite.node)
                ite = prev[ite]
            return path
        #Remove from frontier
        visited[curState] = curState.g
        #Loop through neighbors
        curNode = curState.node
        cur_g = curState.g
        cur_objs = curState.objs[:]
        for node in maze.getNeighbors(curNode[0], curNode[1]):
            new_g = cur_g + 1
            new_objs = cur_objs[:]
            if node in cur_objs:
                new_objs.remove(node)
            reached_state = state(node, new_g, new_objs)
            new_f = new_g + reached_state.getH_corner(Heu_Corner)
            flag = 0
            for i in range(len(frontier)):
                if frontier[i][1] == reached_state:
                    if new_g < frontier[i][1].g:
                        frontier[i] = (new_f, reached_state)
                        prev[reached_state] = curState
                        heapq.heapify(frontier)
                    flag = 1
                    break
            if flag:
                continue

            for x in visited:
                if x == reached_state:
                    flag = 1
                    break
            if flag:
                continue
            heapq.heappush(frontier, (new_f, reached_state))
            prev[reached_state] = curState


def bfs_in_two(maze, node1, node2):
    #Find a single point
    objs = [node1]
    start = node2
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
                    path.insert(0, curPt)
                    curPt = visited[curPt]
                break
            for p in maze.getNeighbors(curPt[0], curPt[1]):  # See neighbors
                if p not in visited and not maze.isWall(curPt[0], curPt[1]):
                    queue.append(p)
                    # Unlike dijkstra, which considered as visited when expanded
                    visited[p] = curPt
        total_path += path
        start = obj
    return path, len(total_path)


#A disjoint set structure we will need
#Guided by the C++ version in CS225
class dset:
    def __init__(self):
        self.upTree = {}

    def addelement(self, node):
        self.upTree[node] = -1

    def find(self, node):  # Return the root
        if type(self.upTree[node]) == int and self.upTree[node] < 0:
            return node
        root = self.find(self.upTree[node])
        self.upTree[node] = root  # Path compression
        return root

    def setunion(self, a, b):
        rootA = self.find(a)
        rootB = self.find(b)
        sizeA = self.upTree[rootA]
        sizeB = self.upTree[rootB]
        newSize = sizeA + sizeB
        if sizeA <= sizeB:
            self.upTree[rootB] = rootA
            self.upTree[rootA] = newSize
        else:
            self.upTree[rootA] = rootB
            self.upTree[rootB] = newSize

    def size(self, elem):
        root = self.find(elem)
        return -1*self.upTree[root]

def Compute_Path(objs, maze):
    heuristic_table = {}
    edge_table = {}
    objs.append(maze.getStart())
    for i in range(len(objs)):
        for j in range(len(objs)):
            if i != j:
                path, path_length = bfs_in_two(
                    maze, objs[i], objs[j])
                heuristic_table[objs[i], objs[j]] = path_length
                edge_table[objs[i], objs[j]] = path
    return heuristic_table, edge_table

def Find_Mst(unvisited_obj, MST_table, heuristic_table):
    #Find MST using Kruskal's algorithm
    if not unvisited_obj:
        return 0
    if frozenset(unvisited_obj) in MST_table:
        return MST_table[frozenset(unvisited_obj)]
    path_cost = PriorityQueue()
    total_path_cost = 0
    path_collection = []
    node_set = dset()
    #Build up the Priority queue and disjoint set
    for i in range(len(unvisited_obj)):
        node_set.addelement(unvisited_obj[i])
        for j in range(i+1, len(unvisited_obj)):
            dist = heuristic_table[(unvisited_obj[i], unvisited_obj[j])] - 2
            path_cost.put((dist, (unvisited_obj[i], unvisited_obj[j])))
    #Compute MST using customed disjoint set data structure
    while node_set.size(unvisited_obj[0]) != len(unvisited_obj):
        cur_path = path_cost.get()
        node0 = list(cur_path[1])[0]
        node1 = list(cur_path[1])[1]
        if node_set.find(node0) == node_set.find(node1):
            continue
        node_set.setunion(node0, node1)
        total_path_cost += cur_path[0]
        path_collection.append(cur_path[1])
    #Cache Value
    #Cache total_path_cost
    MST_table[frozenset(unvisited_obj)] = total_path_cost
    return total_path_cost

def astar_multi(maze):
    frontier = []
    visited = {}  # Mapping from state to g
    prev = {}  # State to state, keep track of the previous state

    objs = maze.getObjectives()
    start = maze.getStart()

    MST_table = {}  # Store mst total length
    heuristic_table, edge_table = Compute_Path(objs[:], maze)
    #Push the starting state
    initState = state(start, 0, objs)
    f_init = Find_Mst(objs, MST_table, heuristic_table)
    heapq.heappush(frontier, (f_init, initState))
    prev[initState] = None

    while True:
        #Pop off the state with the smallest f(g + h)
        cur_state = heapq.heappop(frontier)[1]

        #Reach goal state, start backtracing
        if cur_state.isGoal():
            path = []
            goals_list = []
            ite = cur_state
            while ite:
                goals_list.append(ite.node)
                ite = prev[ite]
            goals_list.reverse()
            for i in range(len(goals_list) - 1):
                path += edge_table[(goals_list[i], goals_list[i+1])][::-1][1:]
            path.insert(0, start)
            path[::-1]
            #print(path)
            return path
        #Remove from frontier
        visited[cur_state] = cur_state.g
        #Loop through neighbors
        cur_node = cur_state.node
        cur_g = cur_state.g
        cur_objs = cur_state.objs[:]
        for node in cur_state.objs:
            new_g = cur_g + heuristic_table[cur_node, node] - 1
            new_objs = cur_objs[:]
            if node in cur_objs:
                new_objs.remove(node)
            reached_state = state(node, new_g, new_objs)
            new_f = new_g + Find_Mst(new_objs,MST_table,heuristic_table)
            #Check visited for repetition
            heapq.heappush(frontier, (new_f, reached_state))
            prev[reached_state] = cur_state
    return []

def extra(maze):
    """
    Runs extra credit suggestion.
    @param maze: The maze to execute the search on.
    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []