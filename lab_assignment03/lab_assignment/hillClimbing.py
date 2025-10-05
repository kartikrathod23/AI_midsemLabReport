from lab3_gen import generate_k_sat_problem
import random

class Node:
    def __init__(self, state):
        self.state = state

#Heuristic Functions 
def heuristic_value_1(clauses, node):
    """Count satisfied clauses."""
    count = 0
    for clause in clauses:
        for literal in clause:
            if (literal > 0 and node.state[literal - 1] == 1) or (literal < 0 and node.state[abs(literal) - 1] == 0):
                count += 1
                break
    return count

def heuristic_value_2(clauses, node):
    """Count satisfied literals (less strict)."""
    state = node.state
    return sum(state[abs(lit) - 1] == 1 for clause in clauses for lit in clause)

#Helper Functions 
def check(clauses, node):
    """Return True if all clauses are satisfied."""
    return heuristic_value_1(clauses, node) == len(clauses)

def gen_successors(node, clauses):
    """Generate neighbor states and pick the best one."""
    best_val = -1
    best_node = node

    for i in range(len(node.state)):
        temp = node.state.copy()
        temp[i] = 1 - temp[i]
        new_node = Node(state=temp)
        val = heuristic_value_2(clauses, new_node)
        if val > best_val:
            best_val = val
            best_node = new_node

    return None if best_node.state == node.state else best_node

#Hill Climb Algorithm
def hill_climb(clauses, k, m, n, max_iter=1000):
    node = Node([0] * n)
    for i in range(max_iter):
        if check(clauses, node):
            print(f"✅ Solution found at iteration {i}: {node.state}")
            return True
        node = gen_successors(node, clauses)
        if node is None:
            print("⚠️ Local minima reached.")
            return False
    return False

#Penetrance Measurement
def calculate_penetrance(num_instances, k, m, n):
    solved = 0
    for _ in range(num_instances):
        clauses = generate_k_sat_problem(k, m, n)
        if hill_climb(clauses, k, m, n):
            solved += 1
    return (solved / num_instances) * 100

if __name__ == "__main__":
    clause = generate_k_sat_problem(3, 50, 50)
    print("Sample Problem:", clause)
    print("Penetrance:", calculate_penetrance(10, 3, 50, 50))
