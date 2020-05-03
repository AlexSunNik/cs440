def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 
    """
    start = maze.getStart()
    queue = []
    visited = {}
    queue.append(start)
    visited[start] = None

    while len(queue) != 0:
        curPt = queue.pop(0)
        if maze.isObjective(curPt[0],curPt[1]):
            path = []
            while curPt is not None:
                path.insert(0, curPt)
                curPt = visited[curPt]
            print(path)
            return path

        for p in maze.getNeighbors(curPt[0], curPt[1]):  # See neighbors
            if p not in visited and not maze.isWall(curPt[0], curPt[1]):
                queue.append(p)
                visited[p] = curPt
    return None