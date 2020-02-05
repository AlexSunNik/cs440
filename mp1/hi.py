def astar(maze):
    # TODO: Write your code here
    # return path, num_states_explored
    num_of_state = 0
    if len(maze.getObjectives()) == 1:
        start_1 = maze.getStart()
        end_1 = maze.getObjectives()[0]
        return cost_sofar(maze, start_1, end_1)

    start = maze.getStart()
    goals_left = maze.getObjectives()
    goals_left.insert(0, start)
    edge_list = {}
    heuristic_list = {}
    # building graph for mst
    for i in goals_left:
        for j in goals_left:
            if i != j:
                construct_path = cost_sofar(maze, i, j)[0]
                edge_list[(i, j)] = construct_path
                heuristic_list[(i, j)] = len(construct_path)
                num_of_state += 10
    not_visited_list = {}
    visited = {}
    cur_path = queue.PriorityQueue()
    mst_weights = get_MST(maze, goals_left, heuristic_list)
    start_r, start_c = maze.getStart()
    start_state = ctor(start_r, start_c, 0, mst_weights)
    start_state.not_visited = maze.getObjectives()

    cur_path.put(start_state)
    not_visited_list[(start_r, start_c)] = len(start_state.not_visited)

    while len(goals_left):
        cur_state = cur_path.get()
        if not cur_state.not_visited:
            break
        for n in cur_state.not_visited:
            n_row, n_col = n
            n_cost = cur_state.cost + \
                heuristic_list[(cur_state.position, n)] - 1
            next_state = ctor(n_row, n_col, n_cost, 0)
            next_state.prev = cur_state
            next_state.not_visited = deepcopy(cur_state.not_visited)
            if n in next_state.not_visited:
                next_state.not_visited.remove(n)
            visited[(n_row, n_col)] = 0
            not_visited_list[n] = len(next_state.not_visited)
            mst_weights = get_MST(maze, cur_state.not_visited, heuristic_list)
            next_state.tcost = n_cost + mst_weights
            a = len(goals_left) - 1
            if a:
                next_state.tcost += len(next_state.not_visited)
            cur_path.put(next_state)
    ret_path1 = print_path(maze, edge_list, cur_state, visited)
    return ret_path1, num_of_state
    
    def astar(maze):
        # TODO: Write your code here
        # return path, num_states_explored
    num_of_state = 0
    if len(maze.getObjectives()) == 1:
        start_1 = maze.getStart()
        end_1 = maze.getObjectives()[0]
        return cost_sofar(maze, start_1, end_1)

    start = maze.getStart()
    goals_left = maze.getObjectives()
    goals_left.insert(0, start)
    edge_list = {}
    heuristic_list = {}
    # building graph for mst
    for i in goals_left:
        for j in goals_left:
            if i != j:
                construct_path = cost_sofar(maze, i, j)[0]
                edge_list[(i, j)] = construct_path
                heuristic_list[(i, j)] = len(construct_path)
                num_of_state += 10
    not_visited_list = {}
    visited = {}
    cur_path = queue.PriorityQueue()
    mst_weights = get_MST(maze, goals_left, heuristic_list)
    start_r, start_c = maze.getStart()
    start_state = ctor(start_r, start_c, 0, mst_weights)
    start_state.not_visited = maze.getObjectives()

    cur_path.put(start_state)
    not_visited_list[(start_r, start_c)] = len(start_state.not_visited)

    while len(goals_left):
        cur_state = cur_path.get()
        if not cur_state.not_visited:
            break
        for n in cur_state.not_visited:
            n_row, n_col = n
            n_cost = cur_state.cost + \
                heuristic_list[(cur_state.position, n)] - 1
            next_state = ctor(n_row, n_col, n_cost, 0)
            next_state.prev = cur_state
            next_state.not_visited = deepcopy(cur_state.not_visited)
            if n in next_state.not_visited:
                next_state.not_visited.remove(n)
            visited[(n_row, n_col)] = 0
            not_visited_list[n] = len(next_state.not_visited)
            mst_weights = get_MST(maze, cur_state.not_visited, heuristic_list)
            next_state.tcost = n_cost + mst_weights
            a = len(goals_left) - 1
            if a:
                next_state.tcost += len(next_state.not_visited)
            cur_path.put(next_state)
    ret_path1 = print_path(maze, edge_list, cur_state, visited)
    return ret_path1, num_of_state


def print_path(maze, path, state, visited):
    ret_path = []
    goals_list = []
    while state:
        goals_list.append(state.position)
        state = state.prev
    total_dot = len(goals_list)-1
    for i in range(total_dot):
        ret_path += path[(goals_list[i], goals_list[i+1])][:-1]
    start = maze.getStart()
    ret_path.append(start)
    ret_path[::-1]
    return ret_path


def get_MST(maze, goals, heuristic_list):
    # Prim
    if not len(goals):
        return 0
    start = goals[0]
    visited = {}
    visited[start] = True
    MST_edges = []
    mst_weights = 0
    while len(visited) < len(goals):
        qe = queue.PriorityQueue()
        for v in visited:
            for n in goals:
                if visited.get(n) == True:
                    continue
                new_edge = (v, n)
                new_cost = heuristic_list[new_edge]-2
                qe.put((new_cost, new_edge))
        add_edge = qe.get()
        MST_edges.append(add_edge[1])
        mst_weights += add_edge[0]
        visited[add_edge[1][1]] = True
    return mst_weights


def cost_sofar(maze, start, end):
    pq = queue.PriorityQueue()
    visited = {}
    result_row, result_col = end
    start_row, start_col = start
    cost = abs(start_row-result_row) + abs(start_col - result_col)
    pq.put((cost, [(start_row, start_col)]))
    while not pq.empty():
        cur_path = pq.get()[1]
        cur_row, cur_col = cur_path[-1]
        if (cur_row, cur_col) in visited:
            continue
        cur_cost = abs(cur_row - result_row) + \
            abs(cur_col - result_col) + len(cur_path) - 1
        visited[(cur_row, cur_col)] = cur_cost
        if (cur_row, cur_col) == (result_row, result_col):
            return cur_path, len(visited)
        for item in maze.getNeighbors(cur_row, cur_col):
            new_cost = abs(item[0] - result_row) + \
                abs(item[1] - result_col) + len(cur_path) - 1
            if item not in visited:
                pq.put((new_cost, cur_path + [item]))
            else:
                # if a node that’s already in the explored set found, test to see if the new h(n)+g(n) is smaller than the old one.
                if visited[item] > new_cost:
                    visited[item] = new_cost
                    pq.put((new_cost, cur_path + [item]))
    return [], 0
