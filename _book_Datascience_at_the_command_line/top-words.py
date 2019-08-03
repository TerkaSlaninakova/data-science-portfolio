import re
import sys
from collections import Counter

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} file.txt NUM_WORDS")
    exit()

file = sys.argv[1]
num_words = int(sys.argv[2])

try:
    with open(file, 'r') as f:
        text = f.read().lower()
        words = re.split('\W+', text)
        cnt = Counter(words)
    for word, count in cnt.most_common(num_words):
        print(f"{word} : {count}")
except FileNotFoundError:
    print(f"{file} does not exist.")