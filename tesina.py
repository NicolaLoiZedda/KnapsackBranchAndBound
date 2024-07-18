import argparse
import time
import sys
import heapq
from collections import deque

DEBUG = False

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
            self.nodes = 0

    def read_from_file(self, file_path):
        with open(file_path, 'r') as file:
            first_line = file.readline().split()
            self.n = int(first_line[0])
            self.capacity = int(first_line[1])
            self.profits = []
            self.weights = []
            
            for _ in range(self.n):
                profit, weight = map(int, file.readline().split())
                self.profits.append(profit)
                self.weights.append(weight)
        
        assert len(self.profits) == len(self.weights) == self.n, "The number of profits and weights must match the specified number of items"
        self.x = [0] * self.n
        self.z = 0
        self.nodes = 0

    def solve_problem(self, search_algorithm, bound_method):
        self.preprocessing()
        self.branch_and_bound(search_algorithm, bound_method)
        return self.x, self.z
    
    def preprocessing(self):
        if DEBUG:
            print('start preprocessing')
        items = [(self.profits[i], self.weights[i], i) for i in range(self.n)]
        items.sort(key=lambda x: x[0] / x[1], reverse=True)        
        
        self.profits = [items[i][0] for i in range(self.n)]
        self.weights = [items[i][1] for i in range(self.n)]
        if DEBUG:
            print('end preprocessing')
    
    def branch_and_bound(self, search_algorithm, bound_method):
        assert search_algorithm == 0 or search_algorithm == 1, "Search algorithm not defined"
        if DEBUG:
            print('B&B')
        upper_bound, break_item = self.calculate_upper_bound(0, 0, 0, bound_method)
        if search_algorithm == 0: # depth-first search
            self.branch_and_bound_dfs(0, 0, 0, [0] * self.n, bound_method, upper_bound, break_item)
        else: # best bound-first search
            self.branch_and_bound_bbfs(bound_method)

    def branch_and_bound_dfs(self, level, current_weight, current_value, solution, bound_method, upper_bound, break_item):
        self.nodes += 1
        if DEBUG:
            print(f'level {level}')
        
        # if it's a "regular" node calculates the upper bound (unless it already has been)
        if level < self.n-1:        
            if upper_bound is None:
                if DEBUG:
                    print(f'level {level} - calculating upper bound')
                upper_bound, break_item = self.calculate_upper_bound(level, current_weight, current_value, bound_method)
            elif DEBUG:
                print(f'level {level} - bound already calculated ({upper_bound})')
                
            if upper_bound <= self.z:
                if DEBUG:
                    print(f'pruning (upper bound lower than best solution - {upper_bound} <= {self.z}=)')
                return
        
        # if search can't continue after this node, calculate the solution of this branch
        if (level == self.n) or (level == break_item and self.capacity - current_weight < min(self.weights[level:])):
            if current_value > self.z:
                if DEBUG:
                    print(f'{current_value} > {self.z}')
                    print(f'NEW SOLUTION: {current_value}')
                self.z = current_value
                self.x = solution[:]
            elif DEBUG:
                print(f'{current_value} <= {self.z}')
            return

        # x[i] = 1
        if DEBUG:
            print(f'cw={current_weight}, w{level+1}={self.weights[level]}, capacity={self.capacity}')
        if current_weight + self.weights[level] <= self.capacity:
            solution[level] = 1
            if DEBUG:
                print(f'x[{level+1}]=1 explore node')
            self.branch_and_bound_dfs(level + 1, current_weight + self.weights[level],
                                  current_value + self.profits[level], solution, bound_method, upper_bound, break_item)
        elif DEBUG:
            print(f'x[{level+1}]=1 don\'t explore node')

        # x[i] = 0
        if level != self.n-1:
            solution[level] = 0
            if DEBUG:
                print(f'x[{level+1}]=0 explore node')
            self.branch_and_bound_dfs(level + 1, current_weight, current_value, solution, bound_method, None, None)
        
        if DEBUG:
            print(f'-----------backtracking from level {level} to level {level-1}--------------')
    
    def branch_and_bound_bbfs(self, bound_method):
        priority_queue = []

        # initial node (level 0, current_weight 0, current_value 0)
        initial_node = (0, 0, 0, [0] * self.n, float('inf'))
        heapq.heappush(priority_queue, (-initial_node[4], initial_node)) # -initial_node[4] = -inf

        while priority_queue:
            _, node = heapq.heappop(priority_queue)
            level, current_weight, current_value, solution, upper_bound = node

            self.nodes += 1
            if DEBUG:
                print(f'level {level}: current_weight = {current_weight}')

            if level < self.n - 1:
                if upper_bound <= self.z:
                    if DEBUG:
                        print(f'pruning (upper bound lower than best solution - {upper_bound} <= {self.z}=)')
                    continue

            if (level == self.n) or (self.capacity - current_weight < min(self.weights[level:], default=0)):
                if current_value > self.z:
                    if DEBUG:
                        print(f'{current_value} > {self.z}')
                        print(f'NEW SOLUTION: {current_value}')
                    self.z = current_value
                    self.x = solution[:]
                elif DEBUG:
                    print(f'{current_value} <= {self.z}')
                continue

            if current_weight + self.weights[level] <= self.capacity:
                new_solution = solution[:]
                new_solution[level] = 1
                left_upper_bound, _ = self.calculate_upper_bound(level + 1, current_weight + self.weights[level], current_value + self.profits[level], bound_method)
                if DEBUG:
                    print(f'x[{level + 1}]=1 calculate node bound')
                    print(f'left upper bound = {left_upper_bound}')
                heapq.heappush(priority_queue, (-left_upper_bound, (level + 1, current_weight + self.weights[level], current_value + self.profits[level], new_solution, left_upper_bound)))

            new_solution = solution[:]
            new_solution[level] = 0
            right_upper_bound, _ = self.calculate_upper_bound(level + 1, current_weight, current_value, bound_method)
            if DEBUG:
                print(f'x[{level + 1}]=0 calculate node bound')
                print(f'right upper bound = {right_upper_bound}')
            heapq.heappush(priority_queue, (-right_upper_bound, (level + 1, current_weight, current_value, new_solution, right_upper_bound)))
    
    def calculate_upper_bound(self, level, current_weight, current_value, bound_method):
        assert bound_method == 0 or bound_method == 1, "Bound method not defined"
        if DEBUG:
            print(f'level {level} - calculating break item')
        break_item = self.calculate_break_item(level, current_weight)
        if DEBUG:
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
                if DEBUG:
                    print(f'break item = {i+1}')
                return i
        if DEBUG:
            print('no break item')
        return self.n  # all items can be included
    
    def calculate_residual_capacity(self, level, current_weight, break_item):
        w = current_weight
        for i in range(level, break_item):
            w += self.weights[i]
        
        if DEBUG:
            print(f'residual capacity = {self.capacity - w}')
        return self.capacity - w

    def upper_bound_dantzig(self, level, current_weight, current_value, break_item, residual_capacity):
        if DEBUG:
            print(f'current value={current_value}')
        result = current_value
        for i in range(level, break_item):
            if DEBUG:
                print(self.profits[i], end=" ")
            result += self.profits[i]
            
        if DEBUG:
            print('upper bound dantzig')
        if break_item < self.n:
            if DEBUG:
                print(f'{residual_capacity} * {self.profits[break_item]} / {self.weights[break_item]}')
            result += residual_capacity * self.profits[break_item] / self.weights[break_item]
        
        if DEBUG:
            print(f'upper bound = {int(result)}')
        
        return int(result)

    def upper_bound_martello_toth(self, level, current_weight, current_value, break_item, residual_capacity):
        if DEBUG:
            print(f'current value={current_value}')
        res1 = current_value
        res2 = current_value
        
        for i in range(level, break_item):
            if DEBUG:
                print(self.profits[i], end=" ")
            res1 += self.profits[i]
        
        if DEBUG:
            print('upper bound mt')

        if break_item < self.n:
            res1 += residual_capacity * self.profits[break_item] / self.weights[break_item]
            
            for i in range(level, break_item+1):
                if DEBUG:
                    print(self.profits[i], end=" ")
                res2 += self.profits[i]
            if break_item > 0 and break_item < self.n:
                res2 -= (self.weights[break_item] - residual_capacity) * (self.profits[break_item-1] / self.weights[break_item-1])
        
        result = max(int(res1), int(res2))
        
        if DEBUG:
            print(f'upper bound = {result}')
        
        return result

# command-line argument parsing
parser = argparse.ArgumentParser(description='Solve the knapsack problem.')
parser.add_argument('file_path', type=str, help='Path to the file containing profits, weights, and capacity.')
parser.add_argument('search_algorithm', type=int, choices=[0, 1], help='Search algorithm to use: 0 for DFS, 1 for BFS.')
parser.add_argument('bound_method', type=int, choices=[0, 1], help='Bound method to use: 0 for Dantzig, 1 for Martello-Toth.')
args = parser.parse_args()

sys.setrecursionlimit(5000)

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
print(f'Number of nodes: {kp.nodes}')
