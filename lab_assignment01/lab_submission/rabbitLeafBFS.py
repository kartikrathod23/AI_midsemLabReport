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


def swap(state, i, j):
    s = list(state)
    s[i], s[j] = s[j], s[i]
    return tuple(s)


def next_states(state):
    e = state.index(-1)
    moves = [-2, -1, 1, 2]
    results = []

    for m in moves:
        pos = e + m
        if 0 <= pos < 7:
            if m > 0 and state[pos] == 1:
                new_s = swap(state, e, pos)
                if is_valid(new_s):
                    results.append(new_s)
            elif m < 0 and state[pos] == 0:
                new_s = swap(state, e, pos)
                if is_valid(new_s):
                    results.append(new_s)
    return results


def bfs(start, goal):
    q = deque([(start, [])])
    seen = set()
    visited_count = 0
    max_q_size = 0

    while q:
        max_q_size = max(max_q_size, len(q))
        state, path = q.popleft()
        if state in seen:
            continue
        seen.add(state)
        new_path = path + [state]
        visited_count += 1

        if state == goal:
            print(f"Total Number Of Nodes Visited: {visited_count}")
            print(f"Max Size Of Queue: {max_q_size}")
            return new_path

        for nxt in next_states(state):
            q.append((nxt, new_path))
    return None


start = (0, 0, 0, -1, 1, 1, 1)
goal = (1, 1, 1, -1, 0, 0, 0)
sol = bfs(start, goal)

if sol:
    print("Solution found:")
    print(f"Number Of Nodes In Solution: {len(sol)}")
    for step in sol:
        print(step)
else:
    print("No solution found.")
