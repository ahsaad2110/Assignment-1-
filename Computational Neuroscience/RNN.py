import numpy as np

sentences = [
    ["the", "cat", "sat", "on"],
    ["the", "dog", "sat", "under"],
    ["the", "man", "slept", "on"],
    ["a", "woman", "sat", "near"],
    ["the", "child", "played", "outside"]
]

vocab = list(set(word for sentence in sentences for word in sentence))
word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for w, i in word_to_ix.items()}
vocab_size = len(vocab)

def one_hot(ix, size):
    vec = np.zeros((size, 1))
    vec[ix] = 1
    return vec

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def forward(inputs):
    hs = {}
    hs[-1] = np.zeros((hidden_size, 1))
    for t in range(len(inputs)):
        hs[t] = np.tanh(np.dot(Wxh, inputs[t]) + np.dot(Whh, hs[t-1]) + bh)
    y = np.dot(Why, hs[len(inputs)-1]) + by
    p = softmax(y)
    return p, hs

def loss(p, target):
    return -np.sum(target * np.log(p))

hidden_size = 16
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))

learning_rate = 0.1
for epoch in range(3000):
    for sentence in sentences:
        X = [one_hot(word_to_ix[w], vocab_size) for w in sentence[:3]]
        Y = one_hot(word_to_ix[sentence[3]], vocab_size)
        p, hs = forward(X)
        dWhy = np.dot((p - Y), hs[2].T)
        dby = p - Y
        dh = np.dot(Why.T, dby)
        for t in reversed(range(3)):
            dtanh = (1 - hs[t] * hs[t]) * dh
            dWxh = np.dot(dtanh, X[t].T)
            dWhh = np.dot(dtanh, hs[t-1].T)
            dbh = dtanh
            Wxh -= learning_rate * dWxh
            Whh -= learning_rate * dWhh
            bh -= learning_rate * dbh
            dh = np.dot(Whh.T, dtanh)
        Why -= learning_rate * dWhy
        by -= learning_rate * dby

def predict(word1, word2, word3):
    if word1 not in word_to_ix or word2 not in word_to_ix or word3 not in word_to_ix:
        return "Unknown word(s)"
    X = [one_hot(word_to_ix[word1], vocab_size),
         one_hot(word_to_ix[word2], vocab_size),
         one_hot(word_to_ix[word3], vocab_size)]
    p, _ = forward(X)
    return ix_to_word[np.argmax(p)]

while True:
    inp = input("Enter 3 words separated by space (or 'exit'): ").strip().lower()
    if inp == "exit":
        break
    words = inp.split()
    if len(words) != 3:
        print("Please enter exactly 3 words.")
        continue
    print("Predicted 4th word:", predict(words[0], words[1], words[2]))