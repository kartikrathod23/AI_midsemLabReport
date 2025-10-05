import heapq
import random
from collections import defaultdict
import spacy
import string
from spacy.cli import download
import nltk
from nltk.tokenize import sent_tokenize


# ----------- File & Text Processing -----------

def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def extract_sentences(text):
    text = text.lower()
    nlp = spacy.load("en_core_web_sm")
    return [sent.text for sent in nlp(text).sents]


def remove_punctuation(sentence):
    return sentence.translate(str.maketrans("\n", " ", string.punctuation))


def process_text_file(file_path):
    content = read_file(file_path)
    sentences = extract_sentences(content)
    cleaned_sentences = [remove_punctuation(sentence) for sentence in sentences]
    return cleaned_sentences


# ----------- Global Documents -----------

document1 = []
document2 = []


# ----------- Node Definition -----------

class Node:
    def __init__(self, state, parent=None, g=0, h=0, w1=1, w2=1):
        self.state = state          # (index_doc1, index_doc2, move)
        self.parent = parent
        self.g = g                  # cost from start
        self.h = h                  # heuristic estimate
        self.f = w1 * g + w2 * h    # total cost

    def __lt__(self, other):
        return self.f < other.f


# ----------- Cost Functions -----------

def get_difference(index_doc1=None, index_doc2=None):
    """Character-level difference between two sentences."""
    sentence1 = document1[index_doc1] if index_doc1 is not None else ""
    sentence2 = document2[index_doc2] if index_doc2 is not None else ""

    chars1, chars2 = list(sentence1), list(sentence2)
    count1 = defaultdict(int)

    for c in chars1:
        count1[c] += 1

    diff = max(len(chars1) - len(chars2), 0)

    for c in chars2:
        if count1[c] == 0:
            diff += 1
        else:
            count1[c] -= 1

    return diff


def distance(state, goal_state):
    """Heuristic: sum of differences from current to goal alignment."""
    i1, i2, _ = state
    g1, g2, _ = goal_state
    dist = 0

    while i1 <= g1 or i2 <= g2:
        if i1 <= g1 and i2 <= g2:
            dist += get_difference(i1, i2)
        elif i1 <= g1:
            dist += get_difference(i1, None)
        elif i2 <= g2:
            dist += get_difference(None, i2)
        i1 += 1
        i2 += 1

    return dist


def char_level_edit_distance(s1, s2):
    """Classic dynamic programming edit distance."""
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = (
                dp[i - 1][j - 1]
                if s1[i - 1] == s2[j - 1]
                else min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
            )

    return dp[n][m]


def edit_distance(state, goal_state):
    """Compute cost based on move type."""
    i1, i2, move = state

    if move == 0:  # Alignment
        return char_level_edit_distance(document1[i1 - 1], document2[i2 - 1])
    elif move == 1:  # Insertion
        return len(document2[i2 - 1])
    elif move == 2:  # Deletion
        return len(document1[i1 - 1])

    return 0


# ----------- Successor Generation -----------

def get_successors(node):
    """Generate possible moves: alignment, insertion, deletion."""
    moves = [(1, 1, 0), (0, 1, 1), (1, 0, 2)]
    i1, i2, _ = node.state

    successors = []
    for di1, di2, mv in moves:
        new_state = (i1 + di1, i2 + di2, mv)
        successors.append(Node(new_state, node))
    return successors


# ----------- A* Algorithm -----------

def a_star(start_state, goal_state):
    start_node = Node(start_state)
    goal_node = Node(goal_state)

    open_list = []
    heapq.heappush(open_list, (start_node.f, start_node))
    visited = set()
    explored = 0

    while open_list:
        _, node = heapq.heappop(open_list)
        if tuple(node.state) in visited:
            continue
        visited.add(tuple(node.state))
        explored += 1

        if (node.state[0] == goal_node.state[0] + 1 and
            node.state[1] == goal_node.state[1] + 1):
            path = []
            while node:
                path.append(node.state)
                node = node.parent
            print("Total nodes explored:", explored)
            return path[::-1]

        for succ in get_successors(node):
            if succ.state[0] <= goal_node.state[0] + 1 and succ.state[1] <= goal_node.state[1] + 1:
                succ.g = node.g + edit_distance(succ.state, goal_node.state)
                succ.h = distance(succ.state, goal_node.state)
                succ.f = succ.g + succ.h
                heapq.heappush(open_list, (succ.f, succ))

    print("Total nodes explored:", explored)
    return None


# ----------- Document Alignment -----------

def align_documents(states, start_state, goal_state):
    aligned = []
    for s in states:
        if s[:2] == start_state[:2]:
            continue
        if s[-1] == 0:
            aligned.append(document1[s[0] - 1])
        elif s[-1] == 1:
            aligned.append(document2[s[1] - 1])
        elif s[-1] == 2:
            continue

        if s[:2] == goal_state[:2]:
            print("Goal state reached")
    return aligned


# ----------- Utilities -----------

def count_words(sentence):
    return len(sentence.split())


# ----------- Main Execution -----------

if __name__ == "__main__":
    document1 = process_text_file("doc1.txt")
    document2 = process_text_file("doc2.txt")

    start_state = (0, 0, 0)
    goal_state = (len(document1) - 1, len(document2) - 1, 0)

    print("Goal state:", goal_state)
    path = a_star(start_state, goal_state)

    if path:
        aligned_doc = align_documents(path, start_state, goal_state)
        print("\nAligned Document:", aligned_doc)

        total_words = sum(count_words(s) for s in document1)
        print("\nDocument 1 word count:", total_words)

        for i in range(min(len(aligned_doc), len(document2))):
            ed = char_level_edit_distance(aligned_doc[i], document2[i])
            print(f"\nSentence comparison:\n  Doc3: {aligned_doc[i]}\n  Doc2: {document2[i]}")
            print(f"  Edit Distance: {ed}")
