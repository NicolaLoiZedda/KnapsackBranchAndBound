import argparse
import time
from collections import deque

class KnapsackProblem:
    def __init__(self, profits=None, weights=None, capacity=None, file_path=None):
        if file_path:
            self.read_from_file(file_path)
        else:
            self.profits = profits
            self.weights = weights
            assert len(profits) == len(weights), "Profits and weights must have the same length"
            self.n = len(profits)
            self.capacity = capacity
            self.x = [0] * self.n
            self.z = 0

    def read_from_file(self, file_path):
        with open(file_path, 'r') as file:
            first_line = file.readline().split()
            self.n = int(first_line[0])
            self.capacity = float(first_line[1])
            self.profits = []
            self.weights = []
            
            for _ in range(self.n):
                profit, weight = map(float, file.readline().split())
                self.profits.append(profit)
                self.weights.append(weight)
        
        assert len(self.profits) == len(self.weights) == self.n, "The number of profits and weights must match the specified number of items"
        self.x = [0] * self.n
        self.z = 0

    def solve_problem(self, search_algorithm, bound_method):
        self.preprocessing()
        self.branch_and_bound(search_algorithm, bound_method)
        return self.x, self.z
    
    def preprocessing(self):
        print('start preprocessing')
        items = [(self.profits[i], self.weights[i], i) for i in range(self.n)]
        items.sort(key=lambda x: x[0] / x[1], reverse=True)        
        
        self.profits = [items[i][0] for i in range(self.n)]
        self.weights = [items[i][1] for i in range(self.n)]
        print('end preprocessing')
    
    def branch_and_bound(self, search_algorithm, bound_method):
        assert search_algorithm == 0 or search_algorithm == 1, "Search algorithm not defined"
        print('B&B')
        if search_algorithm == 0: # depth-first search
            self.branch_and_bound_dfs(0, 0, 0, [0] * self.n, bound_method, None, None)
        else: # breadth-first search
            self.branch_and_bound_bfs(bound_method)

    def branch_and_bound_dfs(self, level, current_weight, current_value, solution, bound_method, upper_bound, break_item):
        print(f'level {level}')
        
        if level < self.n-1:        
            if upper_bound is None:
                print(f'level {level} - calculating upper bound')
                upper_bound, break_item = self.calculate_upper_bound(level, current_weight, current_value, bound_method)
            else:
                print(f'level {level} - bound already calculated ({upper_bound})')
                
            if upper_bound <= self.z:
                print(f'pruning (upper bound lower than best solution - {upper_bound} <= {self.z}=)')
                return
        
        if (level == self.n) or (level == break_item and self.capacity - current_weight < min(self.weights[level:])):
            if current_value > self.z:
                print(f'{current_value} > {self.z}')
                print(f'NEW SOLUTION: {current_value}')
                self.z = current_value
                self.x = solution[:]
            else:
                print(f'{current_value} <= {self.z}')
            return

        # x[i] = 1
        print(f'cw={current_weight}, w{level+1}={self.weights[level]}, capacity={self.capacity}')
        if current_weight + self.weights[level] <= self.capacity:
            solution[level] = 1
            print(f'x[{level+1}]=1 explore node')
            self.branch_and_bound_dfs(level + 1, current_weight + self.weights[level],
                                  current_value + self.profits[level], solution, bound_method, upper_bound, break_item)
        else:
            print(f'x[{level+1}]=1 don\'t explore node')

        # x[i] = 0        
        if level != self.n-1:
            solution[level] = 0
            print(f'x[{level+1}]=0 explore node')
            self.branch_and_bound_dfs(level + 1, current_weight, current_value, solution, bound_method, None, None)
            
        print(f'-----------backtracking from level {level} to level {level-1}--------------')
    
    def branch_and_bound_bfs(self, bound_method):
        queue = deque([(0, 0, 0, [0] * self.n)])  # (level, current_weight, current_value, solution)

        while queue:
            level, current_weight, current_value, solution = queue.popleft()

            if level == self.n:
                if current_value > self.z:
                    self.z = current_value
                    self.x = solution[:]
                continue

            # x[i] = 1
            if current_weight + self.weights[level] <= self.capacity:
                print(f'x[{level+1}]=1 explore node')
                new_solution = solution[:]
                new_solution[level] = 1
                queue.append((level + 1, current_weight + self.weights[level],
                              current_value + self.profits[level], new_solution))
            else:
                print(f'x[{level+1}]=1 don\'t explore node')

            # x[i] = 0
            upper_bound = self.calculate_upper_bound(level, current_weight, current_value, bound_method)
            if upper_bound > self.z:
                print(f'x[{level+1}]=0 explore node')
                new_solution = solution[:]
                new_solution[level] = 0
                queue.append((level + 1, current_weight, current_value, new_solution))
            else:
                print('pruning')

    def calculate_upper_bound(self, level, current_weight, current_value, bound_method):
        assert bound_method == 0 or bound_method == 1, "Bound method not defined"
        print(f'level {level} - calculating break item')
        break_item = self.calculate_break_item(level, current_weight)
        print(f'level {level} - calculating residual capacity')
        residual_capacity = self.calculate_residual_capacity(level, current_weight, break_item)
        if bound_method == 0: # Dantzig upper bound
            return self.upper_bound_dantzig(level, current_weight, current_value, break_item, residual_capacity), break_item
        else: # Martello-Toth upper bound
            return self.upper_bound_martello_toth(level, current_weight, current_value, break_item, residual_capacity), break_item
    
    def calculate_break_item(self, level, current_weight):
        w = current_weight
        for i in range(level, self.n):
            w += self.weights[i]
            if w > self.capacity:
                print(f'break item = {i+1}')
                return i
        print('no break item')
        return self.n  # all items can be included
    
    def calculate_residual_capacity(self, level, current_weight, break_item):
        w = current_weight
        for i in range(level, break_item):
            w += self.weights[i]
        
        print(f'residual capacity = {self.capacity - w}')
        return self.capacity - w

    def upper_bound_dantzig(self, level, current_weight, current_value, break_item, residual_capacity):
        print(f'current value={current_value}')
        result = current_value
        for i in range(level, break_item):
            print(self.profits[i], end=" ")
            result += self.profits[i]
            
        print('upper bound dantzig')
        if break_item < self.n:
            print(f'{residual_capacity} * {self.profits[break_item]} / {self.weights[break_item]}')
            result += residual_capacity * self.profits[break_item] / self.weights[break_item]
        
        print(f'upper bound = {int(result)}')
        
        return int(result)

    def upper_bound_martello_toth(self, level, current_weight, current_value, break_item, residual_capacity):
        res1 = current_value
        for i in range(level, break_item):
            res1 += self.profits[i]
        res1 += residual_capacity * (self.profits[break_item] / self.weights[break_item])
        
        print('upper bound mt')
        
        res2 = current_value
        for i in range(level, break_item+1):
            res2 += self.profits[i]
        res2 -= (self.weights[break_item] - residual_capacity) * (self.profits[break_item-1] / self.weights[break_item-1])
        
        result = max(int(res1), int(res2))
        
        print(f'upper bound = {result}')
        
        return result

# command-line argument parsing
parser = argparse.ArgumentParser(description='Solve the knapsack problem.')
parser.add_argument('file_path', type=str, help='Path to the file containing profits, weights, and capacity.')
parser.add_argument('search_algorithm', type=int, choices=[0, 1], help='Search algorithm to use: 0 for DFS, 1 for BFS.')
parser.add_argument('bound_method', type=int, choices=[0, 1], help='Bound method to use: 0 for Dantzig, 1 for Martello-Toth.')
args = parser.parse_args()

# define the knapsack problem
kp = KnapsackProblem(file_path=args.file_path)

# start measuring execution time
start_time = time.time()

# solve the knapsack problem
x, z = kp.solve_problem(args.search_algorithm, args.bound_method)

# stop measuring execution time
end_time = time.time()
execution_time = end_time - start_time

# print results
print(f'x = {x}')
print(f'z = {z}')
print(f'Execution time: {execution_time} seconds')
