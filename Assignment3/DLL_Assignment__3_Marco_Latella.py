import nltk
from nltk.tokenize import *
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def download_files(url):
    os.system(f" wget {url}")


def print_file_data(file_name):
    nltk.download('punkt')
    with open(file_name, "r") as fp:
        lines = []
        for c, line in enumerate(fp):
            lines.append(line)
        print(f"Lines: {c + 1}")

        lines_lengths = [len(line) for line in lines]
        lines_avg = np.around(np.mean(lines_lengths), 2)
        print(f"Lines length avg: {lines_avg}")

    file = open(file_name, "r")
    data = file.read()

    sentences = sent_tokenize(data)
    print(f"Sentences: {len(sentences)}")

    # Are considered words also "," "." and so on.
    words = word_tokenize(data)
    print(f"Words: {len(words)}")

    chars = [c for c in data]
    # Are considered Characters also whitespaces, new line, etc.
    print(f"Characters: {len(chars)}")


class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>'):
        self.id_to_string = {}
        self.string_to_id = {}

        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0

        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1

    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt file and generate a 1D PyTorch tensor
# containing the whole text mapped to sequence of token IDs, and a vocab object.
class LongTextData:

    def __init__(self, file_path, vocab=None, extend_vocab=True, device='cuda'):
        self.data, self.vocab = self.text_to_data(file_path, vocab, extend_vocab, device)

    def __len__(self):
        return len(self.data)

    def text_to_data(self, text_file, vocab, extend_vocab, device):
        """Read a raw text file and create its tensor and the vocab.

        Args:
          text_file: a path to a raw text file.
          vocab: a Vocab object
          extend_vocab: bool, if True extend the vocab
          device: device

        Returns:
          Tensor representing the input text, vocab file

        """
        assert os.path.exists(text_file)
        if vocab is None:
            vocab = Vocabulary()

        data_list = []

        # Construct data
        full_text = []
        print(f"Reading text file from: {text_file}")
        with open(text_file, 'r') as text:
            for line in text:
                tokens = list(line)
                for token in tokens:
                    # get index will extend the vocab if the input
                    # token is not yet part of the text.
                    full_text.append(vocab.get_idx(token, extend_vocab=extend_vocab))

        # convert to tensor
        data = torch.tensor(full_text, device=device, dtype=torch.int64)
        print("Done.")

        return data, vocab


# Since there is no need for schuffling the data, we just have to split
# the text data according to the batch size and bptt length.
# The input to be fed to the model will be batch[:-1]
# The target to be used for the loss will be batch[1:]
class ChunkedTextData:

    def __init__(self, data, bsz, bptt_len, pad_id):
        self.batches = self.create_batch(data, bsz, bptt_len, pad_id)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def create_batch(self, input_data, bsz, bptt_len, pad_id):
        """Create batches from a TextData object .

        Args:
          input_data: a TextData object.
          bsz: int, batch size
          bptt_len: int, bptt length
          pad_id: int, ID of the padding token

        Returns:
          List of tensors representing batches

        """
        batches = []  # each element in `batches` is (len, B) tensor
        text_len = len(input_data)  # number of characters
        segment_len = text_len // bsz + 1  # length of segment or number of batches

        # Question: Explain the next two lines!
        padded = input_data.data.new_full((segment_len * bsz,), pad_id)
        padded[:text_len] = input_data.data
        padded = padded.view(bsz, segment_len).t()
        num_batches = segment_len // bptt_len + 1

        for i in range(num_batches):
            # Prepare batches such that the last symbol of the current batch
            # is the first symbol of the next batch.
            if i == 0:
                # Append a dummy start symbol using pad token
                batch = torch.cat(
                    [padded.new_full((1, bsz), pad_id),
                     padded[i * bptt_len:(i + 1) * bptt_len]], dim=0)
                batches.append(batch)
            else:
                batches.append(padded[i * bptt_len - 1:(i + 1) * bptt_len])
        return batches


class RNNModel(nn.Module):

    def __init__(self, hidden_size, num_layers, size_dict, size_embed_vecs, batch_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.size_dict = size_dict
        self.size_embed_vecs = size_embed_vecs
        self.batch_size = batch_size

        self.embed = nn.Embedding(size_dict, size_embed_vecs)  # size_dict, size_ebed_vecs
        self.lstm = nn.LSTM(size_embed_vecs, hidden_size, num_layers)  # in/out shape: (seq, batch, feature)
        self.linear = nn.Linear(hidden_size, size_dict)

    def forward(self, x, h0, c0):
        # input shape: (len, B, dim)
        # output shape: (len * B, num_classes)

        output = self.embed(x)
        output, (h0, c0) = self.lstm(output, (h0, c0))
        output = self.linear(output)
        return output, (h0.detach(), c0.detach())


# Trainer
class Trainer:

    def __init__(self, model, loss_fn, optimizer, my_data):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.my_data = my_data

    def print_prediction(self, phrase, num_chars, sampling):
        predicted_phrase = self.predict(self.my_data.vocab, phrase, sampling, num_chars)
        print(phrase + predicted_phrase)
        print("-----------------------")

    def train(self, batches, num_pred_chars, sampling, phrase):
        steps = 0
        perp = 2

        while perp >= 1.03:

            h0 = torch.zeros(self.model.num_layers, self.model.batch_size, self.model.lstm.hidden_size).to(device)
            c0 = torch.zeros(self.model.num_layers, self.model.batch_size, self.model.lstm.hidden_size).to(device)

            for i, batch in enumerate(batches):
                self.model.train()
                self.optimizer.zero_grad()

                output, (h0, c0) = self.model(batch[:-1], h0, c0)
                loss = self.loss_fn(output.flatten(end_dim=-2), batch[1:].flatten())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                perp = torch.exp(loss)
                self.optimizer.step()

                if (steps % 150 == 0):
                    print(f"Step: {steps} > Perplexity: {perp}")
                    self.print_prediction(phrase, num_pred_chars, sampling)

                steps += 1

        print(f"Step: {steps} > Perplexity: {perp}")
        self.print_prediction(phrase, num_pred_chars, sampling)

    def pick_char(self, sampling, distrib):
        if not sampling:
            values, token = torch.topk(distrib, k=1, dim=-1)
        else:
            token = torch.multinomial(distrib, num_samples=1)

        return token

    def predict(self, vocab, phrase, sampling, num_chars):
        text = []
        for token in phrase:
            text.append([vocab.get_idx(token)])

        data = torch.tensor(text, device=device, dtype=torch.int64)
        func = nn.Softmax(dim=1)

        h0 = torch.zeros(1, 1, self.model.lstm.hidden_size).to(device)
        c0 = torch.zeros(1, 1, self.model.lstm.hidden_size).to(device)

        with torch.no_grad():
            self.model.eval()
            predicted_phrase = ""

            output, (h0, c0) = self.model(data, h0, c0)

            distrib = func(output[-1])
            token = self.pick_char(sampling, distrib)

            predicted_phrase += vocab.id_to_string[token[0].item()]

            for i in range(num_chars):
                output, (h0, c0) = self.model(token, h0, c0)
                distrib = func(output[-1])
                token = self.pick_char(sampling, distrib)
                predicted_phrase += vocab.id_to_string[token[0].item()]

        return predicted_phrase


def run_task(bptt_length, learning_rate, txt_file, phrase):
    batch_size = 32

    my_data = LongTextData(txt_file, device=device)
    batches = ChunkedTextData(my_data, batch_size, bptt_length, pad_id=0)

    size_vocab = len(my_data.vocab)
    num_classes = size_vocab
    size_embed_vecs = 64

    # Hyper-params:
    hidden_size = 2048
    num_layers = 1

    model = RNNModel(hidden_size, num_layers, size_vocab, size_embed_vecs, batch_size)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    engine = Trainer(model, loss_fn, optimizer, my_data)
    engine.train(batches, num_pred_chars=100, sampling=False, phrase=phrase)

    return [engine, my_data]


def run_combinations(sentence):
    lr = [0.002, 0.00075]
    bptt = [32, 128]

    for l in lr:
        for b in bptt:
            print(f"\n#####\nCombination > Learning rate: {l}, bptt: {b}\n#####\n")
            run_task(b, l, "49010-0.txt", sentence)


def task_1_2_3():
    my_data = LongTextData("49010-0.txt", device=device)
    print("LongTextData __len__: ", my_data.__len__())


def task_1_2_4():
    my_data = LongTextData("49010-0.txt", device=device)
    batches = ChunkedTextData(my_data, 32, 64, pad_id=0)
    print("ChunkedTextData __len__: ", batches.__len__())


def task_1_3_7(engine, my_data):
    input_phrase = "THE HARES AND THE FROGS"
    num_chars = 310
    predicted_phrase = engine.predict(my_data.vocab, input_phrase, False, num_chars)
    print("\n\n" + input_phrase + predicted_phrase)

    input_phrase = "THE RAT AND THE ALLIGATOR"
    num_chars = 260
    predicted_phrase = engine.predict(my_data.vocab, input_phrase, False, num_chars)
    print("\n\n" + input_phrase + predicted_phrase)

    input_phrase = "The rats are pretty"
    num_chars = 127
    predicted_phrase = engine.predict(my_data.vocab, input_phrase, False, num_chars)
    print("\n\n" + input_phrase + predicted_phrase)

    input_phrase = "When I was young I "
    num_chars = 100
    predicted_phrase = engine.predict(my_data.vocab, input_phrase, False, num_chars)
    print("\n\n" + input_phrase + predicted_phrase)


def task_1_3_8(engine, my_data):
    input_phrase = "THE HARES AND THE FROGS"
    num_chars = 100
    predicted_phrase = engine.predict(my_data.vocab, input_phrase, True, num_chars)
    print("\n\n" + input_phrase + predicted_phrase)

    input_phrase = "THE RAT AND THE ALLIGATOR"
    num_chars = 100
    predicted_phrase = engine.predict(my_data.vocab, input_phrase, True, num_chars)
    print("\n\n" + input_phrase + predicted_phrase)


# Support method to cut the original file in a small version
def cut_file():
    file = open("pg925.txt", "r")
    data = file.read()
    cut_len = int(len(data) / 4.)
    data = data[:cut_len]
    file = open("pg925.txt", "w")
    file.write(data)


def task_1_3_9():
    engine, my_data = run_task(64, 0.001, "pg925.txt", "I am the president of ")

    # Input is a starting sentence present in the file
    input_phrase = "I am again called upon by the voice "
    num_chars = 100
    predicted_phrase = engine.predict(my_data.vocab, input_phrase, True, num_chars)
    print("\n\n" + input_phrase + predicted_phrase)

    # This is a sentence not present in the file
    input_phrase = "I have to reduce "
    num_chars = 100
    predicted_phrase = engine.predict(my_data.vocab, input_phrase, True, num_chars)
    print("\n\n" + input_phrase + predicted_phrase)

    # Particular sentence not in file but interesting
    input_phrase = "I love my country, it is the "
    num_chars = 100
    predicted_phrase = engine.predict(my_data.vocab, input_phrase, True, num_chars)
    print("\n\n" + input_phrase + predicted_phrase)


if __name__ == '__main__':
    fables_file = "https://www.gutenberg.org/files/49010/49010-0.txt"
    speeches = "https://www.gutenberg.org/cache/epub/925/pg925.txt"

    download_files(fables_file)
    download_files(speeches)

    # Task 1.1.1
    print_file_data("49010-0.txt")

    # Task 1.2.3
    task_1_2_3()

    # Task 1.2.4
    task_1_2_4()

    # Task 1.3.1
    # The model has been implemented above

    # Task 1.3.2
    # The function required is predict() implemented in the Trainer class

    # Task 1.3.4
    # The train function has been implemented in the Trainer Class

    # Task 1.3.5
    engine, my_data = run_task(64, 0.001, "49010-0.txt", "Dogs likes best to")

    # Task 1.3.6
    run_combinations("Dogs likes best to")

    # Task 1.3.7
    task_1_3_7(engine, my_data)

    # Task 1.3.8
    task_1_3_8(engine, my_data)

    # Task 1.3.9
    cut_file()
    task_1_3_9()

