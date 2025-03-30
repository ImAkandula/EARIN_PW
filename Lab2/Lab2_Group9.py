import copy

class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solution = None
        self.viz = []

    def print_sudoku(self, puzzle):
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - -")
            row_output = ""
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    row_output += " |"
                row_output += f" {puzzle[i][j]}"
            print(row_output.strip())

    def visualize(self):
        print("\nStep-by-step visualization:")
        for step in self.viz:
            print(f"Assigned {step[2]} to ({step[0]}, {step[1]})")

    def forward_checking(self, var, value, assignment):
        temp_domains = copy.deepcopy(self.domains)
        for peer in self.constraints[var]:
            if peer in temp_domains and value in temp_domains[peer]:
                temp_domains[peer].remove(value)
                if not temp_domains[peer]:  # Domain wiped out
                    return None
        return temp_domains



    def is_consistent(self, var, value, assignment):
        for neighbor in self.constraints[var]:
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        return True

    def select_unassigned_variable(self, assignment):
        # Can be optimized with MRV heuristic
        for var in self.variables:
            if var not in assignment:
                return var
        return None

    def backtrack(self, assignment):
        if len(assignment) == len(self.variables):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.domains[var]:
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                self.viz.append((var[0], var[1], value))
                temp_domains = self.forward_checking(var, value, assignment)
                if temp_domains is not None:
                    old_domains = self.domains
                    self.domains = temp_domains
                    result = self.backtrack(assignment)
                    if result:
                        return result
                    self.domains = old_domains
                del assignment[var]
        return None

    def solve(self):
        assignment = {}
        self.solution = self.backtrack(assignment)
        return self.solution


# ---------- SETUP -------------
puzzle = [[5, 3, 0, 0, 7, 0, 0, 0, 0],
          [0, 5, 0, 1, 0, 5, 0, 0, 0],
          [0, 9, 8, 0, 0, 0, 0, 6, 0],
          [0, 0, 0, 0, 0, 3, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 6],
          [0, 0, 0, 0, 0, 0, 2, 8, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 8],
          [0, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 4, 0, 0]]

variables = [(i, j) for i in range(9) for j in range(9) if puzzle[i][j] == 0]

# Constraints map
constraints = {}
for i, j in variables:
    peers = set()
    peers.update(((i, y) for y in range(9) if y != j))  # row
    peers.update(((x, j) for x in range(9) if x != i))  # column
    box_x, box_y = 3 * (i // 3), 3 * (j // 3)
    for x in range(box_x, box_x + 3):
        for y in range(box_y, box_y + 3):
            if (x, y) != (i, j):
                peers.add((x, y))
    constraints[(i, j)] = peers

# Initial domain pruning
domains = {}
for i, j in variables:
    used = set()
    used.update(puzzle[i])  # row
    used.update(puzzle[x][j] for x in range(9))  # column
    box_x, box_y = 3 * (i // 3), 3 * (j // 3)
    used.update(puzzle[x][y] for x in range(box_x, box_x+3) for y in range(box_y, box_y+3))
    domains[(i, j)] = [d for d in range(1, 10) if d not in used]

csp = CSP(variables, domains, constraints)
sol = csp.solve()

solution = [row[:] for row in puzzle]
if sol:
    for (i, j), val in sol.items():
        solution[i][j] = val
    csp.visualize()
    print('*' * 7, 'Solution', '*' * 7)
    csp.print_sudoku(solution)
    
else:
    print("Solution does not exist")
