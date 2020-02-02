#A state representation
#Includes the current node, g_score, and obj. list
class state:
    def __init__(self, curNode, g, objs):
        self.node = curNode
        self.g = g
        self.objs = objs[:] #A deep copy

    def __eq__(self,other):
        if self.node == other.node and set(self.objs) == set(other.objs):
            return True
        else:
            return False

    def isGoal(self):
        return len(self.objs) == 0

    def getH_corner(self,heuristic):
        return heuristic(self.node,self.objs)
    
def astar_corner(maze):
    #state representation in the class "state"

    frontier = {}   #Mapping from state to f
    #visited = {}    #Mapping from state to g
    prev = {}   #State to state, keep track of the previous state

    objs = maze.getObjectives()
    start = maze.getStart()

    #Push the starting state
    initState  = state(start,0,objs)
    f_init = Heu_Corner(start,objs)
    frontier[initState] = f_init
    prev[initState] = None

    while True:
        curState = min(frontier.items(),key=lambda x: x[1])[0]
        if(curState.isGoal()):
            #Reach the goal state
            #Backtrace to get the path
            path = []
            ite = curState
            while ite is not None:
                path.insert(0,ite.node)
                ite = prev[ite]
            return  path
        #Loop through neighbors
        curNode = curState.node
        cur_g = curState.g
        cur_objs = curState.objs[:]
        for node in maze.getNeighbors(curNode):
            new_g = cur_g + 1
            if maze.isObjective(node):
                new_objs = cur_objs.remove(node)
            else:
                new_objs = cur_objs
            reached_state = state(node,new_g,new_objs)
            frontier[reached_state] = reached_state.getH_corner(Heu_Corner)
            prev[reached_state] = curState


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.
        
    @param maze: The maze to execute the search on.
        
    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    frontier = {}  # Frontier list
    visited = {}  # Explored list
    total_path = []  # Total path

    start = maze.getStart()
    objs = maze.getObjectives()  # Get dests
    prev = {}  # Previous values
    frontier[start] = [0, Heu_Corner(start, objs)]
    path_start = start  # Keep track of the start of the current path
    prev[start] = None

    while len(frontier) != 0:
        #Get min f_val pair
        min_pair = min(frontier.items(), key=lambda x: x[1][1])  # Get by f_val
        curNode = min_pair[0]
        del frontier[curNode]
        visited[curNode] = min_pair[1]

        if curNode in objs:
            path = []  # Current path
            ite = curNode  # Iterator for prev
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
            if node in visited:
                continue
            new_g = min_pair[1][0] + 1
            # Tie-breaking here is important
            if new_g < frontier.get(node, [sys.maxsize, sys.maxsize])[1]:
                frontier[node] = [new_g, new_g + Heu_Corner(node, objs)]
                prev[node] = curNode
    return []

    unvisited_obj = []  # Unvisited objectives
    visited_obj = []  # Visited objectives
    total_path = []
    Manhatten_table = {}  # Mapping from pair of nodes to Manhatten distance   Look-up table

    objs = maze.getObjectives()
    unvisited_obj = objs[:]
    Manhatten_table = Compute_Manhatten(unvisited_obj)
    start = maze.getStart()
    # Get initial mst length
    mst_length = Find_Mst(unvisited_obj, Manhatten_table)

    frontier = {}
    frontier[start] = [0, Heu_Multi(
        start, unvisited_obj, mst_length)]
    visited = {}
    prev = {}
    prev[start] = None

    while len(frontier) != 0:
        #Get min f_val pair
        min_pair = min(frontier.items(), key=lambda x: x[1][1])  # Get by f_val
        curNode = min_pair[0]
        del frontier[curNode]
        #visited[curNode] = min_pair[1]
        visited[curNode] = min_pair[1]

        if curNode in unvisited_obj:
            path = []  # Current path
            ite = curNode  # Iterator for prev
            while True:
                ite = prev[ite]
                if ite is None:
                    break
                path.insert(0, ite)

            total_path += path
            unvisited_obj.remove(curNode)  # Remove reached objective
            visited_obj.append(curNode)

            if len(unvisited_obj) == 0:
                total_path.append(curNode)
                print(total_path)
                return total_path

            # Update mst length
            mst_length = Find_Mst(unvisited_obj, Manhatten_table)
            prev = {}
            prev[curNode] = None
            frontier = {}  # Frontier list
            visited = {}  # Explored list
            frontier[curNode] = [0, Heu_Multi(
                curNode, unvisited_obj, mst_length)]

        for node in maze.getNeighbors(curNode[0], curNode[1]):
            if node in visited:
                continue
            new_g = min_pair[1][0] + 1
            # Tie-breaking here is important
            frontier[node] = [
                new_g, new_g + Heu_Multi(node, unvisited_obj, mst_length)]
            prev[node] = curNode



def astar_multi:
    unvisited_obj = []  # Unvisited objectives
    visited_obj = []  # Visited objectives
    total_path = []
    Manhatten_table = {}  # Mapping from pair of nodes to Manhatten distance   Look-up table

    objs = maze.getObjectives()
    unvisited_obj = objs[:]
    Manhatten_table = Compute_Manhatten(unvisited_obj)
    start = maze.getStart()
    # Get initial mst length
    mst_length = Find_Mst(unvisited_obj, Manhatten_table)

    frontier = {}
    frontier[start] = [0, Heu_Multi(
        start, unvisited_obj, mst_length)]
    visited = {}
    prev = {}
    prev[start] = None

    while len(frontier) != 0:
        #Get min f_val pair
        min_pair = min(frontier.items(), key=lambda x: x[1][1])  # Get by f_val
        curNode = min_pair[0]
        del frontier[curNode]
        #visited[curNode] = min_pair[1]
        visited[curNode] = min_pair[1]

        if curNode in unvisited_obj:
                path = []  # Current path
                ite = curNode  # Iterator for prev
                while True:
                    ite = prev[ite]
                    if ite is None:
                        break
                    path.insert(0, ite)

                total_path += path
                unvisited_obj.remove(curNode)  # Remove reached objective
                visited_obj.append(curNode)

                if len(unvisited_obj) == 0:
                    total_path.append(curNode)
                    print(total_path)
                    return total_path

                # Update mst length
                mst_length = Find_Mst(unvisited_obj, Manhatten_table)
                prev = {}
                prev[curNode] = None
                frontier = {}  # Frontier list
                visited = {}  # Explored list
                frontier[curNode] = [0, Heu_Multi(
                    curNode, unvisited_obj, mst_length)]

            for node in maze.getNeighbors(curNode[0], curNode[1]):
                if node in visited:
                    continue
                new_g = min_pair[1][0] + 1
                # Tie-breaking here is important
                frontier[node] = [
                    new_g, new_g + Heu_Multi(node, unvisited_obj, mst_length)]
                prev[node] = curNode

