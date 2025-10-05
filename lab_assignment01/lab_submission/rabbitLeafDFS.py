from collections import deque

def is_valid(state):
    e = state.index(-1)
    if state == (1, 1, 1, -1, 0, 0, 0):
        return True
    if e == 0 and state[e+1] == 0 and state[e+2] == 0:
        return False
    if e == 6 and state[e-1] == 1 and state[e-2] == 1:
        return False
    if e == 1 and state[e-1] == 1 and state[e+1] == 0 and state[e+2] == 0:
        return False
    if e == 5 and state[e+1] == 0 and state[e-1] == 1 and state[e-2] == 1:
        return False
    if state[e-1] == 1 and state[e-2] == 1 and state[e+1] == 0 and state[e+2] == 0:
        return False
    return True


def swap_positions(state, i, j):
    s = list(state)
    s[i], s[j] = s[j], s[i]
    return tuple(s)


def next_states(state):
    e = state.index(-1)
    moves = [-2, -1, 1, 2]
    res = []

    for m in moves:
        pos = e + m
        if 0 <= pos < 7:
            if m > 0 and state[pos] == 1:
                new_s = swap_positions(state, e, pos)
                if is_valid(new_s):
                    res.append(new_s)
            elif m < 0 and state[pos] == 0:
                new_s = swap_positions(state, e, pos)
                if is_valid(new_s):
                    res.append(new_s)
    return res


def bfs(start, goal):
    q = deque([(start, [])])
    seen = set()
    visited_nodes = 0
    max_q = 0

    while q:
        max_q = max(max_q, len(q))
        current, path = q.popleft()
        if current in seen:
            continue
        seen.add(current)
        new_path = path + [current]
        visited_nodes += 1

        if current == goal:
            print(f"Total Number Of Nodes Visited: {visited_nodes}")
            print(f"Max Queue Size: {max_q}")
            return new_path

        for nxt in next_states(current):
            q.append((nxt, new_path))
    return None


start = (0, 0, 0, -1, 1, 1, 1)
goal = (1, 1, 1, -1, 0, 0, 0)
solution = bfs(start, goal)

if solution:
    print("Solution found:")
    print(f"Number of Nodes in Solution: {len(solution)}")
    for step in solution:
        print(step)
else:
    print("No solution found.")
