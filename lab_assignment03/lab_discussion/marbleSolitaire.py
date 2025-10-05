import heapq

class SolitaireNode:
    def __init__(self, state, parent=None, g=0, h=0, w1=1, w2=1):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h
        self.f = w1 * g + w2 * h

    def __lt__(self, other):
        return self.f < other.f


def get_possible_moves(state):
    moves = []
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    jump_over = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(7):
        for c in range(7):
            if state[r][c] == 'O':
                for (dr, dc), (jr, jc) in zip(directions, jump_over):
                    end_r, end_c = r + dr, c + dc
                    jump_r, jump_c = r + jr, c + jc
                    if 0 <= end_r < 7 and 0 <= end_c < 7:
                        if state[jump_r][jump_c] == 'O' and state[end_r][end_c] == '0':
                            moves.append((r, c, end_r, end_c))
    return moves


def apply_move(state, move):
    new_state = [row[:] for row in state]
    start_r, start_c, end_r, end_c = move
    jump_r, jump_c = (start_r + end_r) // 2, (start_c + end_c) // 2

    new_state[end_r][end_c] = 'O'
    new_state[start_r][start_c] = '0'
    new_state[jump_r][jump_c] = '0'

    return new_state


# Heuristics
def heuristic_1(state):
    return sum(row.count('O') for row in state)


def heuristic_2(state):
    center = (3, 3)
    total_distance = 0
    for r in range(7):
        for c in range(7):
            if state[r][c] == 'O':
                total_distance += abs(r - center[0]) + abs(c - center[1])
    return total_distance



# Search Algorithms

def best_first_search(initial_state, heuristic_func):
    start_node = SolitaireNode(initial_state)
    open_list = []
    heapq.heappush(open_list, (start_node.h, start_node))
    visited = set()
    max_size = 0

    while open_list:
        _, node = heapq.heappop(open_list)
        max_size = max(max_size, len(open_list))

        state_tuple = tuple(map(tuple, node.state))
        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        if heuristic_func(node.state) == 0:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print(f"Max nodes in queue: {max_size}")
            print(f"Visited nodes: {len(visited)}")
            print(f"Path length: {len(path)}")
            return path[::-1]

        for move in get_possible_moves(node.state):
            new_state = apply_move(node.state, move)
            h = heuristic_func(new_state)
            new_node = SolitaireNode(new_state, node, h=h)
            heapq.heappush(open_list, (new_node.h, new_node))
    return None


def a_star_search(initial_state, heuristic_func):
    start_node = SolitaireNode(initial_state)
    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    visited = set()
    max_size = 0

    while open_list:
        _, node = heapq.heappop(open_list)
        max_size = max(max_size, len(open_list))

        state_tuple = tuple(map(tuple, node.state))
        if state_tuple in visited:
            continue
        visited.add(state_tuple)

        if heuristic_func(node.state) == 0:
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print(f"Max nodes in queue: {max_size}")
            print(f"Visited nodes: {len(visited)}")
            print(f"Path length: {len(path)}")
            return path[::-1]

        for move in get_possible_moves(node.state):
            new_state = apply_move(node.state, move)
            g = node.g + 1
            h = heuristic_func(new_state)
            new_node = SolitaireNode(new_state, node, g=g, h=h)
            heapq.heappush(open_list, (new_node.f, new_node))
    return None


# Main Execution

if __name__ == "__main__":
    start_state = [
        ['-', '-', 'O', 'O', 'O', '-', '-'],
        ['-', '-', 'O', 'O', 'O', '-', '-'],
        ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['O', 'O', 'O', '0', 'O', 'O', 'O'],
        ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
        ['-', '-', 'O', 'O', 'O', '-', '-'],
        ['-', '-', 'O', 'O', 'O', '-', '-']
    ]

    moves = get_possible_moves(start_state)
    for move in moves:
        print(f"Start: {move[0], move[1]} -> End: {move[2], move[3]}")

    print("\nBest-First Search Results:")
    best_first_search(start_state, heuristic_2)

    print("\nA* Search Results:")
    a_star_search(start_state, heuristic_2)
