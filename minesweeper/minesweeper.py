import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) == self.count:
            return set(self.cells)
        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return set(self.cells)
        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Initial height and width
        self.height = height
        self.width = width

        # Keeping track of cells clicked on
        self.moves_made = set()

        # Keeping track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.
        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)
        # 2) mark the cell as safe
        self.safes.add(cell)

        # 3) add a new sentence
        # Get nearby cells
        nearby = self.nearby_cells(cell)[0]
        known_mines = self.nearby_cells(cell)[1]
        count -= known_mines

        new_sentence = Sentence(nearby, count)
        if new_sentence not in self.knowledge:
            self.knowledge.append(new_sentence)

        # 4) mark additional cells as safe/mine
        for sentence in self.knowledge:
            new_safes = sentence.known_safes()
            for safe in new_safes:
                self.mark_safe(safe)
            new_mines = sentence.known_mines()
            for mine in new_mines:
                self.mark_mine(mine)
            
        # 5) add new sentence to knowledge base
        import copy
        known_sentences = copy.deepcopy(self.knowledge)
        for sentence_1 in known_sentences:
            known_sentences.remove(sentence_1)
            for sentence_2 in known_sentences:
                if (len(sentence_1.cells) != 0 and len(sentence_2.cells) != 0):
                    if (len(sentence_1.cells) > len(sentence_2.cells)):
                        big_set = sentence_1.cells
                        small_set = sentence_2.cells
                        diff = sentence_1.count - sentence_2.count
                    elif (len(sentence_2.cells) > len(sentence_1.cells)):
                        big_set = sentence_2.cells
                        small_set = sentence_1.cells
                        diff = sentence_2.count - sentence_1.count
                    elif len(sentence_1.cells) == len(sentence_2.cells):
                        continue
                else:
                    continue
                if big_set >= small_set:
                    diff_set = big_set - small_set
                    if len(diff_set) == 1:
                        new = diff_set.pop()
                        if diff == 0:
                            self.mark_safe(new)
                        elif diff == 1:
                            self.mark_mine(new)
                    else:
                        self.knowledge.append(Sentence(diff_set,diff))
      
    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.
        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for safe in self.safes:
            if safe in self.moves_made:
                continue
            else:
                self.moves_made.add(safe)
                print("Safe move : ",safe)
                return safe

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        moves_list = []
        board = []
        for i in range(self.height):
            for j in range(self.width):
                board.append((i,j))
        for cell in board:
            # if not yet visited and not a mine
            if (cell not in self.mines and cell not in self.moves_made):
                moves_list.append(cell)
        if len(moves_list) != 0:
            random_move = random.choice(moves_list)
            self.moves_made.add(random_move)
            print("Random move : ", random_move)
            return random_move

    def nearby_cells(self, cell):
        """ New function to find the neighbouring cells """
        mines = 0
        i, j = cell
        nearby = set()
        for row in range(i-1, i+2):
            for col in range(j-1, j+2):
                # if position inside board and not cell itself and not yet visited
#                if ((col >= 0 and col < self.width) and (row >= 0 and row < self.height)):
                if (0 <= row < self.height and 0 <= col < self.width):
                    if ((row, col) != cell and (row, col) is not self.moves_made):
                        if (row, col) in self.mines:
                            mines += 1
                        elif (row, col) in self.safes:
                            continue
                        else:
                            nearby.add((row, col))
        return nearby, mines