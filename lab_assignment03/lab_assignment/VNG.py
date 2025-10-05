import random
from lab3_gen import generate_k_sat_problem


class Node:
    def __init__(self, state):
        self.state = state


# Heuristic Functions

def heuristic_value_1(clauses, node):
    count = 0
    for curr_clause in clauses:
        for literal in curr_clause:
            if literal > 0 and node.state[literal - 1] == 1:
                count += 1
                break
            if literal < 0 and node.state[abs(literal) - 1] == 0:
                count += 1
                break
    return count


def heuristic_value_2(clauses, node):
    count = 0
    for curr_clause in clauses:
        for literal in curr_clause:
            if node.state[abs(literal) - 1] == 1:
                count += 1
    return count



# Clause Satisfaction Check

def is_solution(clauses, node):
    if node is None:
        return False

    satisfied = 0
    for curr_clause in clauses:
        for literal in curr_clause:
            if literal > 0 and node.state[literal - 1] == 1:
                satisfied += 1
                break
            if literal < 0 and node.state[abs(literal) - 1] == 0:
                satisfied += 1
                break

    return satisfied == len(clauses)




# Successor Generation Functions

def gen_1(node, clauses):
    max_value = -1
    best_node = node
    unchanged_count = 0

    for i in range(len(node.state)):
        temp = node.state.copy()
        temp[i] = 1 - temp[i]  
        new_node = Node(temp)
        val = heuristic_value_2(clauses, new_node)

        if val > max_value:
            max_value = val
            best_node = new_node
        else:
            unchanged_count += 1

    if unchanged_count == len(node.state):
        return None
    return best_node


def gen_2(node, clauses, num_neighbors=10):
    max_value = -1
    best_node = node

    for _ in range(num_neighbors):
        temp = node.state.copy()
        num_flips = random.choice([1, 2])

        indices = random.sample(range(len(node.state)), num_flips)
        for idx in indices:
            temp[idx] = 1 - temp[idx]

        new_node = Node(temp)
        val = heuristic_value_2(clauses, new_node)

        if val > max_value:
            max_value = val
            best_node = new_node

    if best_node.state == node.state:
        return None
    return best_node


def gen_3(node, clauses, num_neighbors=10):
    max_value = -1
    best_node = node

    for _ in range(num_neighbors):
        temp = node.state.copy()
        num_flips = random.choice([1, 2, 3])

        indices = random.sample(range(len(node.state)), num_flips)
        for idx in indices:
            temp[idx] = 1 - temp[idx]

        new_node = Node(temp)
        val = heuristic_value_2(clauses, new_node)

        if val > max_value:
            max_value = val
            best_node = new_node

    if best_node.state == node.state:
        return None
    return best_node



# Hill Climbing Search

def hill_climb(clauses, node, gen_func, k, m, n, max_iter=1000):
    prev_node = node

    for step in range(max_iter):
        if is_solution(clauses, node):
            print(f"Solution found after {step} steps")
            print(f"Solution: {node.state}")
            return node

        if node is None:
            print("Local minimum reached")
            print(f"Best partial solution: {prev_node.state}")
            return prev_node

        temp_node = gen_func(node, clauses)
        prev_node = node
        node = temp_node

    return node




# Variable-Generation Neighborhood Strategy

def vgn(clauses, k, m, n):
    node = Node([0] * n)

    node = hill_climb(clauses, node, gen_1, k, m, n)
    if is_solution(clauses, node):
        print("Solution found using gen_1")
        return node

    print("Trying gen_2 ...")
    node = hill_climb(clauses, node, gen_2, k, m, n)
    if is_solution(clauses, node):
        print("Solution found using gen_2")
        return node

    print("Trying gen_3 ...")
    node = hill_climb(clauses, node, gen_3, k, m, n)
    if is_solution(clauses, node):
        print("Solution found using gen_3")
        return node

    return is_solution(clauses, node)



# Penetrance Evaluation

def calculate_penetrance(num_instances, k, m, n):
    solved_count = 0

    for _ in range(num_instances):
        clauses = generate_k_sat_problem(k, m, n)
        if vgn(clauses, k, m, n):
            solved_count += 1

    penetrance = (solved_count / num_instances) * 100
    return penetrance




# Main Execution

if __name__ == "__main__":
    clauses = generate_k_sat_problem(3, 75, 75)
    result = calculate_penetrance(20, 3, 10, 10)
    print(f"Penetrance: {result:.2f}%")
