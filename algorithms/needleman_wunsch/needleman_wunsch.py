import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class NeedlemanWunsch:
    def __init__(self, match_score: int = 2, mismatch_penalty: int = -1, gap_penalty: int = -2):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty

        self.seq1 = ""
        self.seq2 = ""
        self.scoring_matrix = None
        self.traceback_matrix = None
        self.traceback_path = []
        self.traceback_method = "standard"

    def _get_score(self, char1: str, char2: str) -> int:
        return self.match_score if char1 == char2 else self.mismatch_penalty

    def _initialize_matrix(self) -> None:
        rows = len(self.seq1) + 1
        cols = len(self.seq2) + 1

        self.scoring_matrix = np.zeros((rows, cols), dtype=int)
        self.traceback_matrix = np.zeros((rows, cols), dtype=int)

        for j in range(1, cols):
            self.scoring_matrix[0, j] = self.scoring_matrix[0, j - 1] + self.gap_penalty
            self.traceback_matrix[0, j] = 2  # Left (insertion)

        for i in range(1, rows):
            self.scoring_matrix[i, 0] = self.scoring_matrix[i - 1, 0] + self.gap_penalty
            self.traceback_matrix[i, 0] = 1  # Up (deletion)

    def _fill_matrix(self) -> None:
        rows = len(self.seq1) + 1
        cols = len(self.seq2) + 1

        for i in range(1, rows):
            for j in range(1, cols):
                match_score = (self.scoring_matrix[i - 1, j - 1] +
                               self._get_score(self.seq1[i - 1], self.seq2[j - 1]))
                delete_score = self.scoring_matrix[i - 1, j] + self.gap_penalty
                insert_score = self.scoring_matrix[i, j - 1] + self.gap_penalty

                scores = [match_score, delete_score, insert_score]
                max_score = max(scores)
                max_index = scores.index(max_score)

                self.scoring_matrix[i, j] = max_score
                self.traceback_matrix[i, j] = max_index

    def _traceback(self) -> Tuple[str, str]:
        """
        Perform traceback to reconstruct the optimal alignment.

        Traditional way to traceback:
            - First checks for diagonal move (match/mismatch)
            - Then checks for gaps (up/left)
            - Generally prioritizes diagonal matches over gaps
            - Tends to produce alignments with matches spread out
        """
        aligned_seq1 = []
        aligned_seq2 = []
        self.traceback_path = []

        i = len(self.seq1)
        j = len(self.seq2)

        while i > 0 or j > 0:
            self.traceback_path.append((i, j))
            if i > 0 and j > 0 and self.traceback_matrix[i, j] == 0:
                # Diagonal move (match/mismatch)
                aligned_seq1.append(self.seq1[i - 1])
                aligned_seq2.append(self.seq2[j - 1])
                i -= 1
                j -= 1
            elif i > 0 and self.traceback_matrix[i, j] == 1:
                aligned_seq1.append(self.seq1[i - 1])
                aligned_seq2.append('-')
                i -= 1
            else:
                aligned_seq1.append('-')
                aligned_seq2.append(self.seq2[j - 1])
                j -= 1

        self.traceback_path.append((0, 0))
        self.traceback_path.reverse()

        # Reverse sequences since we built them backwards
        return ''.join(reversed(aligned_seq1)), ''.join(reversed(aligned_seq2))

    def _traceback_gap_priority(self) -> Tuple[str, str]:
        """
        Implementation from https://bioboot.github.io/bimm143_W20/class-material/nw/:
            - First checks for gaps (up, then left)
            - Uses diagonal moves as last resort
            - Prioritizes gaps over diagonal moves when scores are equal
            - Tends to produce alignments with gaps grouped together
        """
        aligned_seq1 = []
        aligned_seq2 = []
        self.traceback_path = []

        i = len(self.seq1)
        j = len(self.seq2)

        while i > 0 or j > 0:
            self.traceback_path.append((i, j))

            current_score = self.scoring_matrix[i, j]

            # Check for UP move (deletion in seq2)
            if i > 0 and current_score == self.scoring_matrix[i - 1, j] + self.gap_penalty:
                aligned_seq1.append(self.seq1[i - 1])
                aligned_seq2.append('-')
                i -= 1

            # Check for LEFT move (insertion in seq2)
            elif j > 0 and current_score == self.scoring_matrix[i, j - 1] + self.gap_penalty:
                aligned_seq1.append('-')
                aligned_seq2.append(self.seq2[j - 1])
                j -= 1

            # DIAGONAL move as last resort (match/mismatch)
            else:
                if i > 0 and j > 0:
                    aligned_seq1.append(self.seq1[i - 1])
                    aligned_seq2.append(self.seq2[j - 1])
                    i -= 1
                    j -= 1
                elif i > 0:
                    # Force up move if only i > 0
                    aligned_seq1.append(self.seq1[i - 1])
                    aligned_seq2.append('-')
                    i -= 1
                else:
                    # Force left move if only j > 0
                    aligned_seq1.append('-')
                    aligned_seq2.append(self.seq2[j - 1])
                    j -= 1

        self.traceback_path.append((0, 0))
        self.traceback_path.reverse()

        return ''.join(reversed(aligned_seq1)), ''.join(reversed(aligned_seq2))

    def align(self, seq1: str, seq2: str, traceback_method: str = "standard") -> Tuple[str, str, int]:
        self.seq1 = seq1.upper()
        self.seq2 = seq2.upper()
        self.traceback_method = traceback_method
        self._initialize_matrix()

        self._fill_matrix()

        if traceback_method == "gap_priority":
            aligned_seq1, aligned_seq2 = self._traceback_gap_priority()
        elif traceback_method == "standard":
            aligned_seq1, aligned_seq2 = self._traceback()
        else:
            raise ValueError("traceback_method must be 'standard' or 'gap_priority'")

        final_score = self.scoring_matrix[len(self.seq1), len(self.seq2)]

        return aligned_seq1, aligned_seq2, final_score

    def visualize(self, figsize=(12, 8), cmap='viridis', show_path=True):
        if self.scoring_matrix is None:
            print("No matrix to visualize. Run align() first.")
            return

        plt.figure(figsize=figsize)

        path_mask = np.zeros_like(self.scoring_matrix)
        for i, j in self.traceback_path:
            path_mask[i, j] = 1

        ax = sns.heatmap(self.scoring_matrix, annot=True, fmt='.0f', cmap=cmap,
                         cbar_kws={'label': 'Alignment Score'})

        if show_path and self.traceback_path:
            for idx in range(len(self.traceback_path) - 1):
                i1, j1 = self.traceback_path[idx]
                i2, j2 = self.traceback_path[idx + 1]
                plt.plot([j1 + 0.5, j2 + 0.5], [i1 + 0.5, i2 + 0.5],
                         color='red', linewidth=3, marker='o', markersize=6,
                         markerfacecolor='red', markeredgecolor='darkred')

        plt.xticks(np.arange(len(self.seq2) + 1) + 0.5, ['-'] + list(self.seq2), rotation=0)
        plt.yticks(np.arange(len(self.seq1) + 1) + 0.5, ['-'] + list(self.seq1), rotation=0)

        plt.title(f'Needleman-Wunsch Scoring Matrix with Traceback Path\n'
                  f'Match: {self.match_score}, Mismatch: {self.mismatch_penalty}, Gap: {self.gap_penalty}')
        plt.xlabel('Sequence 2')
        plt.ylabel('Sequence 1')

        ax = plt.gca()
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()

        plt.tight_layout()
        plt.show()

    def print_alignment(self, aligned_seq1: str, aligned_seq2: str, score: int):
        method_name = "Gap-Priority" if self.traceback_method == "gap_priority" else "Standard"
        print(f"\n{method_name} Alignment (Score: {score}):")
        print(f"Seq1: {aligned_seq1}")

        alignment_visual = ""
        for a, b in zip(aligned_seq1, aligned_seq2):
            if a == b:
                alignment_visual += "|"
            elif a == "-" or b == "-":
                alignment_visual += " "
            else:
                alignment_visual += "."

        print(f"      {alignment_visual}")
        print(f"Seq2: {aligned_seq2}")

    @staticmethod
    def get_alignment_stats(aligned_seq1: str, aligned_seq2: str) -> dict:
        matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-')
        mismatches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != b and a != '-' and b != '-')
        gaps = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == '-' or b == '-')

        total_length = len(aligned_seq1)
        identity = (matches / total_length) * 100 if total_length > 0 else 0

        return {
            'matches': matches,
            'mismatches': mismatches,
            'gaps': gaps,
            'total_length': total_length,
            'identity_percentage': identity
        }
