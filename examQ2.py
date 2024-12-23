from typing import List, Dict, Set, Tuple
import random
import numpy as np
from collections import defaultdict, deque
import re


class CrosswordCell:
    def __init__(self, row: int, col: int, is_black: bool = False):
        self.row = row
        self.col = col
        self.is_black = is_black
        self.letter = None
        self.across_word_id = None
        self.down_word_id = None


class CrosswordGrid:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.cells = [
            [CrosswordCell(i, j) for j in range(width)] for i in range(height)
        ]
        self.across_words = {}  # word_id -> (start_cell, length)
        self.down_words = {}  # word_id -> (start_cell, length)
        self.word_dict = set()  # Set of valid words

    def load_dictionary(self, filename: str):
        """Load words from a dictionary file."""
        with open(filename, "r") as f:
            self.word_dict = set(word.strip().upper() for word in f)

    def add_black_cell(self, row: int, col: int):
        """Add a black cell to the grid."""
        self.cells[row][col].is_black = True

    def identify_words(self):
        """Identify all possible word positions in the grid."""
        self.across_words.clear()
        self.down_words.clear()
        word_id = 0

        # Find across words
        for i in range(self.height):
            start_col = 0
            while start_col < self.width:
                if not self.cells[i][start_col].is_black:
                    length = 0
                    while (
                        start_col + length < self.width
                        and not self.cells[i][start_col + length].is_black
                    ):
                        length += 1
                    if length > 1:
                        self.across_words[word_id] = ((i, start_col), length)
                        for j in range(length):
                            self.cells[i][start_col + j].across_word_id = word_id
                        word_id += 1
                start_col += max(1, length)

        # Find down words
        for j in range(self.width):
            start_row = 0
            while start_row < self.height:
                if not self.cells[start_row][j].is_black:
                    length = 0
                    while (
                        start_row + length < self.height
                        and not self.cells[start_row + length][j].is_black
                    ):
                        length += 1
                    if length > 1:
                        self.down_words[word_id] = ((start_row, j), length)
                        for i in range(length):
                            self.cells[start_row + i][j].down_word_id = word_id
                        word_id += 1
                start_row += max(1, length)


class CrosswordSolver:
    def __init__(self, grid: CrosswordGrid):
        self.grid = grid
        self.assignment = {}  # word_id -> word
        self.domains = {}  # word_id -> possible words

    def get_word_pattern(self, word_id: int) -> str:
        """Get the current pattern for a word position."""
        if word_id in self.grid.across_words:
            (row, col), length = self.grid.across_words[word_id]
            pattern = ""
            for j in range(length):
                cell = self.grid.cells[row][col + j]
                if cell.letter:
                    pattern += cell.letter
                else:
                    pattern += "."
            return pattern
        else:
            (row, col), length = self.grid.down_words[word_id]
            pattern = ""
            for i in range(length):
                cell = self.grid.cells[row + i][col]
                if cell.letter:
                    pattern += cell.letter
                else:
                    pattern += "."
            return pattern

    def initialize_domains(self):
        """Initialize domains for all word positions."""
        for word_id in list(self.grid.across_words.keys()) + list(
            self.grid.down_words.keys()
        ):
            _, length = self.grid.across_words.get(word_id) or self.grid.down_words.get(
                word_id
            )
            self.domains[word_id] = {
                word for word in self.grid.word_dict if len(word) == length
            }

    def ac3(self) -> bool:
        """Apply AC3 algorithm for constraint propagation."""
        queue = deque()

        # Add all arcs to queue
        for word1 in self.domains:
            for word2 in self.domains:
                if word1 != word2:
                    queue.append((word1, word2))

        while queue:
            word1, word2 = queue.popleft()
            if self.revise(word1, word2):
                if len(self.domains[word1]) == 0:
                    return False
                for neighbor in self.get_neighbors(word1) - {word2}:
                    queue.append((neighbor, word1))
        return True

    def revise(self, word1: int, word2: int) -> bool:
        """Revise domain of word1 with respect to word2."""
        revised = False
        intersection = self.get_intersection(word1, word2)
        if not intersection:
            return False

        row1, col1, row2, col2 = intersection

        to_remove = set()
        for word in self.domains[word1]:
            valid = False
            for other_word in self.domains[word2]:
                if word[col1] == other_word[col2]:
                    valid = True
                    break
            if not valid:
                to_remove.add(word)
                revised = True

        self.domains[word1] -= to_remove
        return revised

    def get_intersection(self, word1: int, word2: int) -> Tuple[int, int, int, int]:
        """Find intersection point between two words if they cross."""
        if word1 in self.grid.across_words and word2 in self.grid.down_words:
            (row1, col1), _ = self.grid.across_words[word1]
            (row2, col2), _ = self.grid.down_words[word2]

            if (
                row2 <= row1 < row2 + self.grid.down_words[word2][1]
                and col1 <= col2 < col1 + self.grid.across_words[word1][1]
            ):
                return row1 - row2, col2 - col1, col2 - col1, row1 - row2

        elif word1 in self.grid.down_words and word2 in self.grid.across_words:
            return self.get_intersection(word2, word1)

        return None

    def get_neighbors(self, word_id: int) -> Set[int]:
        """Get all words that intersect with the given word."""
        neighbors = set()
        if word_id in self.grid.across_words:
            (row, col), length = self.grid.across_words[word_id]
            for j in range(length):
                cell = self.grid.cells[row][col + j]
                if cell.down_word_id is not None:
                    neighbors.add(cell.down_word_id)
        else:
            (row, col), length = self.grid.down_words[word_id]
            for i in range(length):
                cell = self.grid.cells[row + i][col]
                if cell.across_word_id is not None:
                    neighbors.add(cell.across_word_id)
        return neighbors

    def backtracking_search(self) -> Dict[int, str]:
        """Use backtracking search to find a solution."""
        self.initialize_domains()
        if not self.ac3():
            return None
        return self.backtrack({})

    def backtrack(self, assignment: Dict[int, str]) -> Dict[int, str]:
        """Recursive backtracking algorithm."""
        if len(assignment) == len(self.grid.across_words) + len(self.grid.down_words):
            return assignment

        word_id = self.select_unassigned_variable(assignment)
        for word in self.order_domain_values(word_id, assignment):
            if self.is_consistent(word_id, word, assignment):
                assignment[word_id] = word
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                del assignment[word_id]
        return None

    def select_unassigned_variable(self, assignment: Dict[int, str]) -> int:
        """Select an unassigned variable using MRV heuristic."""
        unassigned = []
        for word_id in self.domains:
            if word_id not in assignment:
                unassigned.append((word_id, len(self.domains[word_id])))
        return min(unassigned, key=lambda x: x[1])[0]

    def order_domain_values(
        self, word_id: int, assignment: Dict[int, str]
    ) -> List[str]:
        """Order domain values using LCV heuristic."""
        return sorted(
            self.domains[word_id],
            key=lambda word: self.count_conflicts(word_id, word, assignment),
        )

    def count_conflicts(
        self, word_id: int, word: str, assignment: Dict[int, str]
    ) -> int:
        """Count number of conflicts word causes with neighboring variables."""
        conflicts = 0
        for neighbor in self.get_neighbors(word_id):
            if neighbor not in assignment:
                intersection = self.get_intersection(word_id, neighbor)
                if intersection:
                    row1, col1, row2, col2 = intersection
                    letter = word[col1]
                    for other_word in self.domains[neighbor]:
                        if other_word[col2] != letter:
                            conflicts += 1
        return conflicts

    def is_consistent(
        self, word_id: int, word: str, assignment: Dict[int, str]
    ) -> bool:
        """Check if assignment is consistent with constraints."""
        # Check if word is already used
        if word in assignment.values():
            return False

        # Check intersections with assigned neighbors
        for neighbor in self.get_neighbors(word_id):
            if neighbor in assignment:
                intersection = self.get_intersection(word_id, neighbor)
                if intersection:
                    row1, col1, row2, col2 = intersection
                    if word[col1] != assignment[neighbor][col2]:
                        return False
        return True

    def apply_solution(self, solution: Dict[int, str]):
        """Apply solution to grid."""
        if not solution:
            return

        for word_id, word in solution.items():
            if word_id in self.grid.across_words:
                (row, col), _ = self.grid.across_words[word_id]
                for j, letter in enumerate(word):
                    self.grid.cells[row][col + j].letter = letter
            else:
                (row, col), _ = self.grid.down_words[word_id]
                for i, letter in enumerate(word):
                    self.grid.cells[row + i][col].letter = letter


def print_grid(grid: CrosswordGrid):
    """Print the crossword grid."""
    for row in grid.cells:
        for cell in row:
            if cell.is_black:
                print("█", end=" ")
            elif cell.letter:
                print(cell.letter, end=" ")
            else:
                print(".", end=" ")
        print()


# Example usage
if __name__ == "__main__":
    # Create a sample grid (3x3 for simpler testing)
    grid = CrosswordGrid(3, 3)

    # Add a single black cell in the center to create a simple cross pattern
    grid.add_black_cell(1, 1)

    # Identify word positions
    grid.identify_words()

    # Add sample words that can definitely form a valid solution
    grid.word_dict = {
        "CAT",
        "DOG",
        "RAT",
        "PIG",
        "BAT",
        "COW",
        "PAW",
        "DAY",
        "CAR",
        "BAR",
        "PAR",
        "TAR",
        "CAN",
        "PAN",
        "TAN",
        "RAN",
    }

    # Create solver and find solution
    solver = CrosswordSolver(grid)
    print("Searching for solution...")
    solution = solver.backtracking_search()

    if solution:
        print("\nSolution found!")
        solver.apply_solution(solution)
        print("\nFinal grid:")
        print_grid(grid)
        print("\nWords used:")
        for word_id, word in solution.items():
            if word_id in grid.across_words:
                print(f"Across {word_id}: {word}")
            else:
                print(f"Down {word_id}: {word}")
    else:
        print("No solution found. Debug information:")
        print(f"Number of across words: {len(grid.across_words)}")
        print(f"Number of down words: {len(grid.down_words)}")
        print(f"Available words of correct lengths: {len(grid.word_dict)}")
        print("Word positions:")
        print("Across words:", grid.across_words)
        print("Down words:", grid.down_words)
