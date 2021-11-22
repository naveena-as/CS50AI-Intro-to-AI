import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v in self.crossword.variables:
            for x in self.crossword.words:
                if len(x) != v.length:
                    self.domains[v].remove(x)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        overlaps = self.crossword.overlaps[x, y]
        if overlaps == None:
            return False
        a, b = overlaps
        to_remove = []
        revised = False
        for X in self.domains[x]:
            overlapping = False
            for Y in self.domains[y]:
                # not same word but same character at intersection
                if X != Y and X[a] == Y[b]:
                    overlapping = True
                    break
            if overlapping == False:
                to_remove.append(X)
        if len(to_remove) != 0:
            for w in to_remove:
                self.domains[x].remove(w)
                revised = True
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = []
            for v in self.crossword.variables:
                for neighbour in self.crossword.neighbors(v):
                    arcs.append((v, neighbour))
        
        for arc in arcs:
            x, y = arc
            if self.revise(x, y):
                # If domain for variable is empty
                if not self.domains[x]:
                    return False
                for neighbour in self.crossword.neighbors(x):
                    arcs.append((neighbour, x))
        return True
        

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for v in self.crossword.variables:
            if v not in assignment.keys() or assignment[v] not in self.crossword.words:
                    return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        for x in assignment.keys():
            # check if every value is of the correct length
            if x.length != len(assignment[x]):
                return False

            for y in assignment.keys()-{x}:
                # check if all values are distinct
                if assignment[x] == assignment[y]:
                    return False
                # check for conflicts between neighboring variables
                overlaps = self.crossword.overlaps[x,y]
                if overlaps != None:
                    a, b = overlaps
                    if assignment[x][a] != assignment[y][b]:
                        return False
        return True
                
    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        neighbors = self.crossword.neighbors(var)
        assigned = assignment.keys()
        result = []
        for word in self.domains[var]:
            eliminated_count = 0
            # consider only unassigned neighbors
            for neighbor in neighbors-assigned:
                (a, b) = self.crossword.overlaps[var, neighbor]
                for _word in self.domains[neighbor]:
                    if word[a] != _word[b]:
                        eliminated_count += 1
            result.append([word, eliminated_count])
        result.sort(key = lambda x: x[1])
        domain_values = []
        for val in result:
            domain_values.append(val[0])
        return domain_values

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned = self.crossword.variables - assignment.keys()
        num_domain = []
        for v in unassigned:
            num_domain.append(len(self.domains[v]))
        minimum = min(num_domain)
        sorted_unassigned = []
        for v in unassigned:
            if len(self.domains[v]) == minimum:
                sorted_unassigned.append(v)
        # ties => return variable with highest degree
        if len(sorted_unassigned) > 1:
            sorted_unassigned = sorted(sorted_unassigned, key = lambda x: len(self.crossword.neighbors(x)))
        return sorted_unassigned[-1]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        # if not a complete assignment yet
        v = self.select_unassigned_variable(assignment)
        for val in self.order_domain_values(v, assignment):
            new = assignment.copy()
            new[v] = val
            # if new value is consistent with the assignment
            if self.consistent(new) and self.inferences(new) == True:
                    result = self.backtrack(new)
                    if result != None:
                        return result
            else:
               return assignment
        return None

    def inferences(self, assignment):
        """ Function to interleave search with inference by maintaining 
        arc consistency every time a new assignment is made """
        arcs = []
        for var in assignment:
            self.domains[var] = {assignment[var]}
            # append (neighbour, variable) pairs to the set of arcs
            for neighbor in self.crossword.neighbors(var):
                arcs.append((neighbor, var))
        # if the new set is arc consistent
        if self.ac3(arcs):
            return True
        return False



def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
