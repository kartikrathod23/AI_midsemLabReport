from collections import deque

def is_valid(state):
    m, c, boat = state
    if not (0 <= m <= 3 and 0 <= c <= 3):
        return False
    if (m > 0 and m < c) or (3 - m > 0 and (3 - m) < (3 - c)):
        return False
    return True

def possible_moves(state):
    m, c, boat = state
    options = [(2, 0), (0, 2), (1, 1), (1, 0), (0, 1)]
    next_states = []
    for dm, dc in options:
        if boat == 1:
            new_state = (m - dm, c - dc, 0)
        else:
            new_state = (m + dm, c + dc, 1)
        if is_valid(new_state):
            next_states.append(new_state)
    return next_states

def bfs(start, goal):
    q = deque([(start, [])])
    seen = set()
    while q:
        current, path = q.popleft()
        if current in seen:
            continue
        seen.add(current)
        new_path = path + [current]
        if current == goal:
            return new_path
        for nxt in possible_moves(current):
            q.append((nxt, new_path))
    return None

start = (3, 3, 1)
goal = (0, 0, 0)
result = bfs(start, goal)

if result:
    print("Solution path:")
    for step in result:
        print(step)
else:
    print("No solution found.")
