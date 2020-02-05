def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    """
    Initial State: Agent at the start state and has not visited any other cities
    Goal State: All other states have been visited
    Gn: Cost of all edges so far
    Heuristic: Manhatten Distance to the nearest unvisited city + total length of MST of unvisited cities
    
    Will a maintain a hash table that stores the MST length for given set of obj
    """
    # TODO: Write your code here
    #Variables needed
    #state representation in the class "state"

    #frontier = {}  # Mapping from state to f
    frontier = []
    visited = {}  # Mapping from state to g
    prev = {}  # State to state, keep track of the previous state

    objs = maze.getObjectives()
    start = maze.getStart()

    MST_table = {}  # Store mst total length
    Manhatten_table = Compute_Manhatten(objs, maze)
    #Push the starting state
    initState = state(start, 0, objs)
    f_init = Heu_Multi(start, objs, Manhatten_table, MST_table, maze)
    heapq.heappush(frontier, (f_init, initState))
    #frontier[initState] = f_init
    prev[initState] = None

    while True:
        curState = heapq.heappop(frontier)[1]

        #Reach goal state, start backtracing
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
        #frontier.pop(curState)
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
            flag = 0
            new_f = new_g + reached_state.getH_multi(
                Heu_Multi, MST_table, Manhatten_table, maze)
            #Check visited for repetition
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
            #frontier[reached_state] = new_f
            prev[reached_state] = curState

    return []
