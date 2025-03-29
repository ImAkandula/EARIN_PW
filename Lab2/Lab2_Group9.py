class CSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solution = None
        self.viz = []  # For visualization steps
 
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
        row, col = var
        temp_domains = {v: self.domains[v][:] for v in self.variables if v not in assignment}
 
        for peer in self.constraints[var]:
            if peer in temp_domains and value in temp_domains[peer]:
                temp_domains[peer].remove(value)
                if not temp_domains[peer]:
                    return None
        return temp_domains
 
    def is_consistent(self, var, value, assignment):
        for neighbor in self.constraints[var]:
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        return True
 
    def select_unassigned_variable(self, assignment):
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
 
# ---------------- SETUP --------------------
puzzle = [[5, 3, 0, 0, 7, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 5, 0, 0, 0],
          [0, 9, 8, 0, 0, 0, 0, 6, 0],
          [0, 0, 0, 0, 0, 3, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0, 0, 6],
          [0, 0, 0, 0, 0, 0, 2, 8, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 8],
          [0, 0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 4, 0, 0]]
 
variables = [(i, j) for i in range(9) for j in range(9) if puzzle[i][j] == 0]
domains = {(i, j): list(range(1, 10)) for i, j in variables}
constraints = {}
 
for i in range(9):
    for j in range(9):
        if puzzle[i][j] == 0:
            peers = set()
            # row and column
            peers.update(((i, y) for y in range(9) if y != j))
            peers.update(((x, j) for x in range(9) if x != i))
            # 3x3 box
            box_x, box_y = 3 * (i // 3), 3 * (j // 3)
            for x in range(box_x, box_x + 3):
                for y in range(box_y, box_y + 3):
                    if (x, y) != (i, j):
                        peers.add((x, y))
            constraints[(i, j)] = peers
 
print('*' * 7, 'Solution', '*' * 7)
csp = CSP(variables, domains, constraints)
sol = csp.solve()
 
solution = [row[:] for row in puzzle]
if sol:
    for (i, j), val in sol.items():
        solution[i][j] = val
 
    csp.print_sudoku(solution)
    csp.visualize()
else:
    print("Solution does not exist")