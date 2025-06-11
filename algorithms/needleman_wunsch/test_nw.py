# Example usage and demonstration
from algorithms.needleman_wunsch.needleman_wunsch import NeedlemanWunsch


def demonstrate_needleman_wunsch():

    aligner = NeedlemanWunsch(match_score=2, mismatch_penalty=-1, gap_penalty=-2)

    seq1 = "ACTGACTGAACCCAA"
    seq2 = "ACTGATCAA"

    print(f"Sequence 1: {seq1}")
    print(f"Sequence 2: {seq2}")

    aligned_seq1, aligned_seq2, score = aligner.align(seq1, seq2, "gap_priority")

    print(f"\nOptimal Alignment (Score: {score}):")
    print(f"{aligned_seq1}")

    # Print alignment with visual indicators
    alignment_visual = ""
    for a, b in zip(aligned_seq1, aligned_seq2):
        if a == b:
            alignment_visual += "|"
        elif a == "-" or b == "-":
            alignment_visual += " "
        else:
            alignment_visual += "."
    print(f"{alignment_visual}")
    print(f"{aligned_seq2}")

    aligner.visualize()
