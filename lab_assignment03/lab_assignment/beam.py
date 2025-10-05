from lab3_gen import generate_k_sat_problem
import random

class Node:
    def __init__(self, state):
        self.state = state

def heuristic_value_1(clauses, node):
    count = 0
    for clause in clauses:
        for lit in clause:
            if (lit > 0 and node.state[lit - 1] == 1) or (lit < 0 and node.state[abs(lit) - 1] == 0):
                count += 1
                break
    return count

def heuristic_value_2(clauses, node):
    return sum(node.state[abs(lit) - 1] == 1 for clause in clauses for lit in clause)

def check(clauses, node):
    return heuristic_value_1(clauses, node) == len(clauses)

def generate_successors(node, clauses, beam_width=3):
    successors = []
    for i in range(len(node.state)):
        temp = node.state.copy()
        temp[i] = 1 - temp[i]
        successors.append(Node(temp))
    successors.sort(key=lambda x: heuristic_value_2(clauses, x), reverse=True)
    return successors[:beam_width]

def gen_successors(node, clauses):
    best_val = -1
    best_node = node
    for i in range(len(node.state)):
        temp = node.state.copy()
        temp[i] = 1 - temp[i]
        new_node = Node(temp)
        val = heuristic_value_1(clauses, new_node)
        if val > best_val:
            best_val = val
            best_node = new_node
    return None if best_node.state == node.state else best_node

def beam_search(clauses, k, m, n, beam_width=3, max_iter=1000):
    node = Node([random.choice([0, 1]) for _ in range(n)])
    if check(clauses, node):
        print(f"✅ Solution found at start: {node.state}")
        return True

    successors = generate_successors(node, clauses, beam_width)
    for i in range(max_iter):
        new_successors = []
        for succ in successors:
            if check(clauses, succ):
                print(f"✅ Found at step {i + 1}: {succ.state}")
                return True
            next_node = gen_successors(succ, clauses)
            if next_node:
                new_successors.append(next_node)
        if not new_successors:
            print("⚠️ Local minima reached.")
            return False
        successors = new_successors
    return False

def calculate_penetrance(num_instances, k, m, n, beam_width=3):
    solved = 0
    for _ in range(num_instances):
        clauses = generate_k_sat_problem(k, m, n)
        if beam_search(clauses, k, m, n, beam_width):
            solved += 1
    return (solved / num_instances) * 100

if __name__ == "__main__":
    print("Beam Search Penetrance (width 3):", calculate_penetrance(10, 3, 25, 25, 3))
    print("Beam Search Penetrance (width 4):", calculate_penetrance(10, 3, 25, 25, 4))
