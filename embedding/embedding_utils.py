import numpy as np

import bcolz
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn

import sys
sys.path.append("../")


glove_path = "./emb_data/glove.6B"


def create_embedding(path: str = glove_path) -> None:
    """Create an embedding based on the small embedding space."""
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(
        np.zeros(1), rootdir=f"{path}/6B.50.dat", mode="w")

    with open(f"{path}/glove.6B.50d.txt", "rb") as f:
        for l in tqdm(f):
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[1:].reshape((400001, 50)), rootdir=f'{path}/6B.50.dat', mode='w')
    vectors.flush()

    pickle.dump(words, open(f"{path}/6B.50_words.pkl", "wb"))
    pickle.dump(word2idx, open(f"{path}/6B.50_idx.pkl", "wb"))


def load_embedding(path: str = glove_path):
    """Reads saved embedding files from local disk."""
    vectors = bcolz.open(f"{path}/6B.50.dat")[:]
    words = pickle.load(open(f"{path}/6B.50_words.pkl", "rb"))
    word2idx = pickle.load(open(f"{path}/6B.50_idx.pkl", "rb"))

    glove = {w: vectors[word2idx[w]] for w in words}

    return glove, (vectors, words, word2idx)


def embedding2weights(glove, words):
    """Converts embedding results."""
    matrix_len = len(words)
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0
    for i, word in tqdm(enumerate(words)):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(50,))
    return weights_matrix


def create_weights(path: str = glove_path) -> np.ndarray:
    """Combined function helps to create weight matrix used in embedding matrix."""
    glove, (_, words, _) = load_embedding(path)
    weights = embedding2weights(glove, words)
    return weights


def create_emb_layer(
    weights_matrix: np.ndarray,
    non_trainable: bool = False
):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    # ==== Alternative Approach ====
    emb_layer.weight = torch.nn.Parameter(torch.Tensor(weights_matrix))
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


if __name__ == "__main__":
    # Testing code
    glove, (vectors, words, word2idx) = load_embedding()
    weights_matrix = create_weights()
    emb_layer, num_embeddings, embedding_dim = create_emb_layer(weights_matrix)
    src = np.random.randint(0, num_embeddings, 100)
    inputs = torch.LongTensor(src)
    emb_results = emb_layer(inputs)
    assert np.all(np.array(emb_results.detach()).astype(np.float32) == weights_matrix[inputs].astype(np.float32))

    def verify_word(word: str = "torch"):
        idx = word2idx[word]
        emb_out = np.array(emb_layer(torch.LongTensor([idx])).detach()).astype(np.float32)
        raw_out = weights_matrix[idx, :].astype(np.float32)
        return np.all(emb_out == raw_out)
    verify_word("python")
