import numpy as np
from typing import List, Tuple


class KnapsackItem:
    def __init__(self, item_id: int, weight: float, value: float):
        self.item_id = item_id
        self.weight = weight
        self.value = value
        self.value_to_weight_ratio = value / weight if weight > 0 else 0

    def __str__(self):
        return f"Item {self.item_id}: weight={self.weight:.2f}, value={self.value:.2f}, ratio={self.value_to_weight_ratio:.2f}"


class KnapsackProblem:
    def __init__(self, items: List[KnapsackItem], capacity: float):
        self.items = items
        self.capacity = capacity
        self.num_items = len(items)

        # Extract weights and values for easier computation
        self.weights = np.array([item.weight for item in items])
        self.values = np.array([item.value for item in items])

        # Calculate problem statistics
        self.total_weight = np.sum(self.weights)
        self.total_value = np.sum(self.values)

    def fitness_function(self, genes: np.ndarray) -> float:
        # Convert to binary (in case of floating point representation)
        selection = (genes > 0.5).astype(int)

        total_weight = np.sum(selection * self.weights)
        total_value = np.sum(selection * self.values)

        # Penalty for exceeding capacity
        if total_weight <= self.capacity:
            # Feasible solution - return actual value
            return total_value
        else:
            # Infeasible solution - much stronger penalty
            excess_weight = total_weight - self.capacity
            excess_ratio = excess_weight / self.capacity

            # Progressive penalty that becomes very harsh for large violations
            if excess_ratio <= 0.1:  # Small violation (≤10%)
                penalty_factor = 0.5
            elif excess_ratio <= 0.2:  # Medium violation (≤20%)
                penalty_factor = 0.8
            else:  # Large violation (>20%)
                penalty_factor = 1.5  # Penalty exceeds value

            penalty = total_value * penalty_factor
            return total_value - penalty

    def decode_solution(self, genes: np.ndarray) -> Tuple[List[int], float, float]:
        selection = (genes > 0.5).astype(int)
        selected_items = [i for i, selected in enumerate(selection) if selected]
        total_weight = np.sum(selection * self.weights)
        total_value = np.sum(selection * self.values)

        return selected_items, total_weight, total_value

    def is_feasible(self, genes: np.ndarray) -> bool:
        selection = (genes > 0.5).astype(int)
        total_weight = np.sum(selection * self.weights)
        return total_weight <= self.capacity

    def print_solution(self, genes: np.ndarray):
        selected_items, total_weight, total_value = self.decode_solution(genes)
        is_feasible = self.is_feasible(genes)

        print("=== Knapsack Solution ===")
        print(f"Total value: {total_value:.2f}")
        print(f"Total weight: {total_weight:.2f} / {self.capacity:.2f}")
        print(f"Weight utilization: {(total_weight / self.capacity) * 100:.1f}%")
        print(f"Feasible: {'Yes' if is_feasible else 'No'}")
        print(f"Number of items selected: {len(selected_items)} / {self.num_items}")

        if selected_items:
            print("\nSelected items:")
            for item_idx in selected_items:
                item = self.items[item_idx]
                print(f"  {item}")
        else:
            print("No items selected")

    def print_problem_info(self):
        print("=== Knapsack Problem Info ===")
        print(f"Number of items: {self.num_items}")
        print(f"Knapsack capacity: {self.capacity:.2f}")
        print(f"Total weight of all items: {self.total_weight:.2f}")
        print(f"Total value of all items: {self.total_value:.2f}")
        print(f"Capacity ratio: {(self.capacity / self.total_weight) * 100:.1f}%")

        print(f"\nAll items:")
        for i in range(self.num_items):
            print(f"  {self.items[i]}")


def create_random_knapsack_problem(
        num_items:int = 100,
        min_value: float = 1.0,
        max_value: float = 100.0,
        min_weight: float = 1.0,
        max_weight: float = 100.0,
        capacity_factor: float = 0.5,
        seed: int = None
) -> KnapsackProblem:
    if seed is not None:
        np.random.seed(seed)

    items = []
    total_weight = 0.0

    for i in range(num_items):
        value = np.random.uniform(min_value, max_value)
        weight = np.random.uniform(min_weight, max_weight)
        items.append(KnapsackItem(item_id=i, value=value, weight=weight))
        total_weight += weight

    # Set capacity to allow approximately capacity_factor of items
    capacity = total_weight * capacity_factor

    return KnapsackProblem(items=items, capacity=capacity)
