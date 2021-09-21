import hnumpy as hnp
import io
import numpy as np
import pathlib
import pickle
import re
import timeit
import torch
import urllib
import zipfile


RAW_EMBEDDINGS_PATH = pathlib.Path("data", "lstm", "wiki-news-300d-1M.vec")
EMBEDDINGS_PATH = pathlib.Path("data", "lstm", "embeddings.pickle")

if not EMBEDDINGS_PATH.exists():
    def load_word_embeddings(file):
        fin = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        embeddings = {
            "indices": {},
            "data": np.zeros((n, d)),
        }
        for i, line in enumerate(fin):
            tokens = line.rstrip().split(' ')
            embeddings["indices"][tokens[0]] = i
            for j, value in enumerate(map(float, tokens[1:])):
                embeddings["data"][i, j] = value

        return embeddings

    if not RAW_EMBEDDINGS_PATH.exists():
        print("\nDownloading embeddings...\n")

        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
        extract_dir = pathlib.Path("data", "lstm")

        zip_path, _ = urllib.request.urlretrieve(url)
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            zip_file.extractall(extract_dir)

    print("Processing embeddings...\n")

    with open(EMBEDDINGS_PATH, "wb") as file:
        pickle.dump(load_word_embeddings(RAW_EMBEDDINGS_PATH), file)
else:
    print()


print("Loading embeddings...\n")

with open(EMBEDDINGS_PATH, "rb") as file:
    embeddings = pickle.load(file)
    embed = lambda tokens: embeddings["data"][[embeddings["indices"][token] for token in tokens], :]

words_to_ignore = []
for word, index in embeddings["indices"].items():
    embedding = embeddings["data"][index, :]
    if embedding.min() < -1 or embedding.max() > 1:
        words_to_ignore.append(word)
for word in words_to_ignore:
    del embeddings["indices"][word]


def encode(sentence):
    sentence = sentence.strip().lower()
    sentence = re.sub(r"[^\w\s]", ' ', sentence)
    sentence = re.sub(r"\s+", ' ', sentence)
    return embed(filter(lambda token: token != "", sentence.split(' ')))


print("Loading dataset...")

DATASET_PATHS = [
    pathlib.Path("data", "lstm", "amazon.txt"),
    pathlib.Path("data", "lstm", "imdb.txt"),
    pathlib.Path("data", "lstm", "yelp.txt"),
]

DATASET = []
for path in DATASET_PATHS:
    with open(path, "r") as file:
        for line in file:
            [line, orientation] = line.strip().split('\t')
            try:
                DATASET.append((encode(line), float(orientation)))
            except:
                pass
assert len(DATASET) > 0

print(len(DATASET), "samples are imported...\n")


HIDDEN_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 10


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=HIDDEN_SIZE)
        self.fc = torch.nn.Linear(HIDDEN_SIZE, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        _, (x, _) = self.lstm(x)
        x = self.fc(x)
        return self.sigmoid(x)


print("Training model for", EPOCHS, "epochs...")

model = Model()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

model.train()
for i in range(EPOCHS):
    for sentence, score in DATASET:
        x = torch.tensor(sentence.reshape(-1, 1, 300))
        prediction = model(x.float())

        optimizer.zero_grad()
        loss = criterion(prediction, torch.tensor([[[score]]]).float())

        loss.backward()
        optimizer.step()

    print("Epoch", i + 1, "is completed...")
model.eval()


class Inferer:
    def __init__(self, model):
        parameters = list(model.lstm.parameters())

        W_ii, W_if, W_ig, W_io = parameters[0].split(HIDDEN_SIZE)
        W_hi, W_hf, W_hg, W_ho = parameters[1].split(HIDDEN_SIZE)
        b_ii, b_if, b_ig, b_io = parameters[2].split(HIDDEN_SIZE)
        b_hi, b_hf, b_hg, b_ho = parameters[3].split(HIDDEN_SIZE)

        self.W_ii = W_ii.detach().numpy()
        self.b_ii = b_ii.detach().numpy()

        self.W_hi = W_hi.detach().numpy()
        self.b_hi = b_hi.detach().numpy()

        self.W_if = W_if.detach().numpy()
        self.b_if = b_if.detach().numpy()

        self.W_hf = W_hf.detach().numpy()
        self.b_hf = b_hf.detach().numpy()

        self.W_ig = W_ig.detach().numpy()
        self.b_ig = b_ig.detach().numpy()

        self.W_hg = W_hg.detach().numpy()
        self.b_hg = b_hg.detach().numpy()

        self.W_io = W_io.detach().numpy()
        self.b_io = b_io.detach().numpy()

        self.W_ho = W_ho.detach().numpy()
        self.b_ho = b_ho.detach().numpy()

        self.W = model.fc.weight.detach().numpy().T
        self.b = model.fc.bias.detach().numpy()

    def infer(self, x):
        x_t, h_t, c_t = None, np.zeros(HIDDEN_SIZE), np.zeros(HIDDEN_SIZE)
        for i in range(x.shape[0]):
            x_t = x[i]
            _, h_t, c_t = self.lstm_cell(x_t, h_t, c_t)

        r = np.dot(h_t, self.W) + self.b
        return self.sigmoid(r)

    def lstm_cell(self, x_t, h_tm1, c_tm1):
        i_t = self.sigmoid(
            np.dot(self.W_ii, x_t) + self.b_ii + np.dot(self.W_hi, h_tm1) + self.b_hi
        )
        f_t = self.sigmoid(
            np.dot(self.W_if, x_t) + self.b_if + np.dot(self.W_hf, h_tm1) + self.b_hf
        )
        g_t = np.tanh(
            np.dot(self.W_ig, x_t) + self.b_ig + np.dot(self.W_hg, h_tm1) + self.b_hg
        )
        o_t = self.sigmoid(
            np.dot(self.W_io, x_t) + self.b_io + np.dot(self.W_ho, h_tm1) + self.b_ho
        )

        c_t = f_t * c_tm1 + i_t * g_t
        h_t = o_t * np.tanh(c_t)

        return o_t, h_t, c_t

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


print("\nCompiling homomorphic inferer...\n")

SENTENCE_LENGTH_LIMIT = 5

inferer = Inferer(model)
homomorphic_inferer = hnp.compile_fhe(
    inferer.infer,
    {
        "x": hnp.encrypted_ndarray(bounds=(-1, 1), shape=(SENTENCE_LENGTH_LIMIT, 300))
    },
    config=hnp.config.CompilationConfig(
        parameter_optimizer="handselected",
        apply_topological_optim=True,
        probabilistic_bounds=6,
    ),
)

context = homomorphic_inferer.create_context()
keys = context.keygen()

operations = homomorphic_inferer.operation_count()
pbses = homomorphic_inferer.pbs_count()

print("\nTarget graph has", operations, "nodes and", pbses, "of them are PBS...")

while True:
    print()

    try:
        query = input(">> ")
    except:
        print()
        break

    if query == "q":
        break

    try:
        embedded = encode(query)
    except KeyError as error:
        print("! the word", error, "is unknown")
        continue

    if embedded.shape[0] > SENTENCE_LENGTH_LIMIT:
        print(f"! the sentence should not contain more than {SENTENCE_LENGTH_LIMIT} tokens")
        continue

    padded = np.zeros((SENTENCE_LENGTH_LIMIT, 300))
    padded[SENTENCE_LENGTH_LIMIT - embedded.shape[0]:, :] = embedded

    original = model(torch.tensor(padded.reshape((-1, 1, 300))).float()).detach().numpy()[0, 0, 0]
    simulated = homomorphic_inferer.simulate(padded)[0]

    start = timeit.default_timer()
    actual = homomorphic_inferer.encrypt_and_run(keys, padded)[0]
    end = timeit.default_timer()

    if actual < 0.35:
        print("- the sentence was negative", end=' ')
    elif actual > 0.65:
        print("+ the sentence was positive", end=' ')
    else:
        print("~ the sentence was neutral", end=' ')

    print(
        f"("
        f"original: {original * 100:.2f}%, "
        f"simulated: {simulated * 100:.2f}%, "
        f"actual: {actual * 100:.2f}%, "
        f"difference: {np.abs(original - actual) * 100:.2f}%, "
        f"took: {end - start:.3f} seconds"
        f")"
    )
