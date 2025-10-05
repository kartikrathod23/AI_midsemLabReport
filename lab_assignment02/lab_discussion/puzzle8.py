import numpy as np
import sys
from time import time


class Node:
    def __init__(self, parent, state, g_cost, h_cost):
        self.parent = parent
        self.state = state
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.total_cost = g_cost + h_cost

    def __hash__(self):
        return hash("".join(self.state.flatten()))

    def __eq__(self, other):
        return hash("".join(self.state.flatten())) == hash("".join(other.state.flatten()))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str(self.state)


class PriorityQueue:
    def __init__(self):
        self.data = []

    def push(self, node):
        self.data.append(node)

    def pop(self):
        if not self.data:
            return None
        idx = min(range(len(self.data)), key=lambda i: self.data[i].total_cost)
        return self.data.pop(idx)

    def is_empty(self):
        return len(self.data) == 0

    def __len__(self):
        return len(self.data)


class Environment:
    def __init__(self, depth, goal_state):
        self.goal_state = goal_state
        self.depth = depth
        self.actions = [1, 2, 3, 4]
        self.start_state = self._generate_start_state()

    def _generate_start_state(self):
        curr_state = self.goal_state
        moves = 0
        while moves < self.depth:
            next_states = self.get_next_states(curr_state)
            choice = np.random.randint(0, len(next_states))
            if np.array_equal(curr_state, next_states[choice]):
                continue
            curr_state = next_states[choice]
            moves += 1
        return curr_state

    def get_next_states(self, state):
        empty_pos = tuple(np.argwhere(state == "_")[0])
        i, j = empty_pos
        states = []

        if i > 0:
            new_state = np.copy(state)
            new_state[i, j], new_state[i - 1, j] = new_state[i - 1, j], new_state[i, j]
            states.append(new_state)
        if i < 2:
            new_state = np.copy(state)
            new_state[i, j], new_state[i + 1, j] = new_state[i + 1, j], new_state[i, j]
            states.append(new_state)
        if j > 0:
            new_state = np.copy(state)
            new_state[i, j], new_state[i, j - 1] = new_state[i, j - 1], new_state[i, j]
            states.append(new_state)
        if j < 2:
            new_state = np.copy(state)
            new_state[i, j], new_state[i, j + 1] = new_state[i, j + 1], new_state[i, j]
            states.append(new_state)

        return states

    def get_start_state(self):
        return self.start_state

    def get_goal_state(self):
        return self.goal_state

    def reached_goal(self, state):
        return np.array_equal(state, self.goal_state)


class Agent:
    def __init__(self, env, heuristic):
        self.env = env
        self.heuristic = heuristic
        self.frontier = PriorityQueue()
        self.explored = {}
        self.goal_node = None
        self.start_state = env.get_start_state()
        self.goal_state = env.get_goal_state()

    def run(self):
        start_node = Node(parent=None, state=self.start_state, g_cost=0, h_cost=0)
        self.frontier.push(start_node)
        steps = 0

        while not self.frontier.is_empty():
            curr = self.frontier.pop()
            if hash(curr) in self.explored:
                continue

            self.explored[hash(curr)] = curr
            if self.env.reached_goal(curr.state):
                self.goal_node = curr
                break

            for next_state in self.env.get_next_states(curr.state):
                h = self.heuristic(next_state, self.goal_state)
                node = Node(curr, next_state, curr.g_cost + 1, h)
                self.frontier.push(node)

            steps += 1

        return steps, self.solution_depth()

    def solution_depth(self):
        node, depth = self.goal_node, 0
        while node:
            depth += 1
            node = node.parent
        return depth

    def print_solution(self):
        node = self.goal_node
        path = []
        while node:
            path.append(node)
            node = node.parent
        for i, n in enumerate(reversed(path), start=1):
            print(f"Step {i}:\n{n}")

    def get_memory(self):
        node_size = 56
        return (len(self.frontier) + len(self.explored)) * node_size


def heuristic0(curr, goal):
    return 0


def heuristic1(curr, goal):
    return np.sum(curr != goal)


def heuristic2(curr, goal):
    dist = 0
    for i in range(3):
        for j in range(3):
            val = curr[i, j]
            pos = np.argwhere(goal == val)[0]
            dist += abs(pos[0] - i) + abs(pos[1] - j)
    return dist


if __name__ == "__main__":
    depth = 500
    goal = np.array([[1, 2, 3], [8, "_", 4], [7, 6, 5]])
    env = Environment(depth, goal)

    print("Start State:\n", env.get_start_state())
    print("Goal State:\n", goal)

    agent = Agent(env, heuristic2)
    agent.run()

    depths = np.arange(0, 501, 50)
    times, memories = {}, {}

    for d in depths:
        avg_time, avg_mem = 0, 0
        for _ in range(50):
            env = Environment(d, goal)
            agent = Agent(env, heuristic2)
            start_t = time()
            agent.run()
            end_t = time()
            avg_time += end_t - start_t
            avg_mem += agent.get_memory()
        times[d] = avg_time / 50
        memories[d] = avg_mem / 50
        print(d, times[d], memories[d])
