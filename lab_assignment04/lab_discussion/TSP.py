import numpy as np
import random
import matplotlib.pyplot as plt



# City Coordinates (Rajasthan)
locations = {
    "Jaipur": (26.9124, 75.7873),
    "Udaipur": (24.5854, 73.7125),
    "Jodhpur": (26.2389, 73.0243),
    "Ajmer": (26.4499, 74.6399),
    "Jaisalmer": (26.9157, 70.9083),
    "Bikaner": (28.0229, 73.3119),
    "Mount Abu": (24.5926, 72.7156),
    "Pushkar": (26.4899, 74.5521),
    "Bharatpur": (27.2176, 77.4895),
    "Kota": (25.2138, 75.8648),
    "Chittorgarh": (24.8887, 74.6269),
    "Alwar": (27.5665, 76.6250),
    "Ranthambore": (26.0173, 76.5026),
    "Sariska": (27.3309, 76.4154),
    "Mandawa": (28.0524, 75.1416),
    "Dungarpur": (23.8430, 73.7142),
    "Bundi": (25.4305, 75.6499),
    "Sikar": (27.6094, 75.1399),
    "Nagaur": (27.2020, 73.7336),
    "Shekhawati": (27.6485, 75.5455),
}

# --------------------------
# Helper Functions
# --------------------------
def euclidean_distance(coord1, coord2):
    """Compute Euclidean distance between two coordinates."""
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def path_cost(tour, distance_matrix):
    """Calculate the total cost of a tour (including return to start)."""
    cost = sum(distance_matrix[tour[i], tour[i + 1]] for i in range(len(tour) - 1))
    cost += distance_matrix[tour[-1], tour[0]]
    return cost

# --------------------------
# Precompute Distance Matrix
# --------------------------
cities = list(locations.keys())
N = len(cities)
D = np.zeros((N, N))

for i in range(N):
    for j in range(i + 1, N):
        dist = euclidean_distance(locations[cities[i]], locations[cities[j]])
        D[i, j] = D[j, i] = dist

# --------------------------
# Simulated Annealing Algorithm
# --------------------------
def simulated_annealing(distance_matrix, max_iter=100000, temp_start=1000, cooling_rate=0.9995):
    """
    Simulated Annealing for TSP
    - distance_matrix: NxN matrix of city distances
    - temp_start: initial temperature
    - cooling_rate: how fast temperature decreases
    - max_iter: number of iterations
    """
    N = len(distance_matrix)

    # Initialize with a random tour
    current_tour = random.sample(range(N), N)
    current_cost = path_cost(current_tour, distance_matrix)

    best_tour = current_tour[:]
    best_cost = current_cost

    cost_history = [current_cost]
    temperature = temp_start

    for iteration in range(max_iter):
        # Generate neighbor by reversing a random segment (2-opt)
        i, j = sorted(random.sample(range(N), 2))
        new_tour = current_tour[:i] + current_tour[i:j+1][::-1] + current_tour[j+1:]

        new_cost = path_cost(new_tour, distance_matrix)
        delta = new_cost - current_cost

        # Acceptance criterion
        if delta < 0 or random.random() < np.exp(-delta / temperature):
            current_tour = new_tour
            current_cost = new_cost

        # Update best solution
        if current_cost < best_cost:
            best_tour, best_cost = current_tour[:], current_cost

        cost_history.append(best_cost)

        # Decrease temperature
        temperature *= cooling_rate
        if temperature < 1e-8:
            break

    return best_tour, best_cost, cost_history

# --------------------------
# Run Algorithm
# --------------------------
best_tour, best_cost, cost_history = simulated_annealing(D)

# --------------------------
# Results
# --------------------------
print("\nBest Tour (City Order):")
print(" → ".join(cities[i] for i in best_tour))
print(f"\nBest Tour Cost: {best_cost:.3f}\n")

# --------------------------
# Visualization
# --------------------------
plt.figure(figsize=(12, 6))

# (a) Route Plot
plt.subplot(1, 2, 1)
tour_coords = np.array([locations[cities[i]] for i in best_tour] + [locations[cities[best_tour[0]]]])
plt.plot(tour_coords[:, 1], tour_coords[:, 0], 'o-', color='royalblue', label="Optimized Tour")
plt.title("Optimized Rajasthan Tour (Simulated Annealing)")
for idx, city_idx in enumerate(best_tour):
    plt.text(tour_coords[idx, 1], tour_coords[idx, 0], cities[city_idx], fontsize=8)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()

# (b) Cost Convergence Plot
plt.subplot(1, 2, 2)
plt.plot(cost_history, color='darkorange')
plt.title("Tour Cost Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Tour Cost")

plt.tight_layout()
plt.show()
