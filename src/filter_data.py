import sys

filename = sys.argv[1]

words = []
stopwords = []
with open(filename) as f:
    for line in f.readlines():
        for word in line.split(' '):
            word = filter(lambda x: x.isalpha(),word.lower())
            tmp = ''
            words.append(tmp.join(list(word)))


print(words)
