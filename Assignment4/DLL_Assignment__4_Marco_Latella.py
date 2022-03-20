import os
import math
import time

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.transformer import Transformer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset


class Vocabulary:

    def __init__(self, pad_token="<pad>", unk_token='<unk>', eos_token='<eos>',
                 sos_token='<sos>'):
        self.id_to_string = {}
        self.string_to_id = {}

        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0

        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1

        # add the default unknown token
        self.id_to_string[2] = eos_token
        self.string_to_id[eos_token] = 2

        # add the default unknown token
        self.id_to_string[3] = sos_token
        self.string_to_id[sos_token] = 3

        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self.sos_id = 3

    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    # if extend_vocab is True, add the new word
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt files and generate parallel text dataset:
# self.data[idx][0] is the tensor of source sequence
# self.data[idx][1] is the tensor of target sequence
# See examples in the cell below.
class ParallelTextDataset(Dataset):

    def __init__(self, src_file_path, tgt_file_path, src_vocab=None,
                 tgt_vocab=None, extend_vocab=False, device='cuda'):
        (self.data, self.src_vocab, self.tgt_vocab, self.src_max_seq_length,
         self.tgt_max_seq_length) = self.parallel_text_to_data(
            src_file_path, tgt_file_path, src_vocab, tgt_vocab, extend_vocab,
            device)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def parallel_text_to_data(self, src_file, tgt_file, src_vocab=None,
                              tgt_vocab=None, extend_vocab=False,
                              device='cuda'):
        # Convert paired src/tgt texts into torch.tensor data.
        # All sequences are padded to the length of the longest sequence
        # of the respective file.

        assert os.path.exists(src_file)
        assert os.path.exists(tgt_file)

        if src_vocab is None:
            src_vocab = Vocabulary()

        if tgt_vocab is None:
            tgt_vocab = Vocabulary()

        data_list = []
        # Check the max length, if needed construct vocab file.
        src_max = 0
        with open(src_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]  # remove line break
                length = len(tokens)
                if src_max < length:
                    src_max = length

        tgt_max = 0
        with open(tgt_file, 'r') as text:
            for line in text:
                tokens = list(line)[:-1]
                length = len(tokens)
                if tgt_max < length:
                    tgt_max = length
        tgt_max += 2  # add for begin/end tokens

        src_pad_idx = src_vocab.pad_id
        tgt_pad_idx = tgt_vocab.pad_id

        tgt_eos_idx = tgt_vocab.eos_id
        tgt_sos_idx = tgt_vocab.sos_id

        # Construct data
        src_list = []
        print(f"Loading source file from: {src_file}")
        with open(src_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)[:-1]
                for token in tokens:
                    seq.append(src_vocab.get_idx(
                        token, extend_vocab=extend_vocab))
                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)
                # padding
                new_seq = var_seq.data.new(src_max).fill_(src_pad_idx)
                new_seq[:var_len] = var_seq
                src_list.append(new_seq)

        tgt_list = []
        print(f"Loading target file from: {tgt_file}")
        with open(tgt_file, 'r') as text:
            for line in tqdm(text):
                seq = []
                tokens = list(line)[:-1]
                # append a start token
                seq.append(tgt_sos_idx)
                for token in tokens:
                    seq.append(tgt_vocab.get_idx(
                        token, extend_vocab=extend_vocab))
                # append an end token
                seq.append(tgt_eos_idx)

                var_len = len(seq)
                var_seq = torch.tensor(seq, device=device, dtype=torch.int64)

                # padding
                new_seq = var_seq.data.new(tgt_max).fill_(tgt_pad_idx)
                new_seq[:var_len] = var_seq
                tgt_list.append(new_seq)

        # src_file and tgt_file are assumed to be aligned.
        assert len(src_list) == len(tgt_list)
        for i in range(len(src_list)):
            data_list.append((src_list[i], tgt_list[i]))

        print("Done.")

        return data_list, src_vocab, tgt_vocab, src_max, tgt_max


# `DATASET_DIR` should be modified to the directory where you downloaded
# the dataset. On Colab, use any method you like to access the data
# e.g. upload directly or access from Drive, ...


class Collector():

    def __init__(self):
        self.task = None
        self.train_set = None
        self.valid_set = None
        self.train_data_loader = None
        self.valid_data_loader = None
        self.src_vocab = None
        self.tgt_vocab = None

    def get_dataset(self, task):
        self.task = task

        DATASET_DIR = "/content"

        TRAIN_FILE_NAME = "train"
        VALID_FILE_NAME = "interpolate"

        INPUTS_FILE_ENDING = ".x"
        TARGETS_FILE_ENDING = ".y"

        TASK = task

        src_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{INPUTS_FILE_ENDING}"
        tgt_file_path = f"{DATASET_DIR}/{TASK}/{TRAIN_FILE_NAME}{TARGETS_FILE_ENDING}"

        print(src_file_path)
        self.train_set = ParallelTextDataset(src_file_path, tgt_file_path, extend_vocab=True)

        # get the vocab
        self.src_vocab = self.train_set.src_vocab
        self.tgt_vocab = self.train_set.tgt_vocab

        src_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{INPUTS_FILE_ENDING}"
        tgt_file_path = f"{DATASET_DIR}/{TASK}/{VALID_FILE_NAME}{TARGETS_FILE_ENDING}"


        self.valid_set = ParallelTextDataset(
            src_file_path, tgt_file_path, src_vocab=self.src_vocab, tgt_vocab=self.tgt_vocab,
            extend_vocab=False)


########
# Taken from:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# or also here:
# https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # shape (max_len, 1, dim)
        self.register_buffer('pe', pe)  # Will not be trained.

    def forward(self, x):
        """Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        assert x.size(0) < self.max_len, (
            f"Too long sequence length: increase `max_len` of pos encoding")
        # shape of x (len, B, dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Encoder(nn.Module):

    def __init__(self, d_model, nhead, dim_ff, nlayers_enc, dropout):
        super(Encoder, self).__init__()

        self.trl = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout)
        self.transformer_enc = nn.TransformerEncoder(self.trl, nlayers_enc)

    def forward(self, x, src_key_padding_mask):
        return self.transformer_enc(x, src_key_padding_mask=src_key_padding_mask)


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, nlayers_dec, dropout):
        super(Decoder, self).__init__()

        self.tdl = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout)
        self.transformer_dec = nn.TransformerDecoder(self.tdl, nlayers_dec)

    def forward(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        return self.transformer_dec(tgt,
                                    memory,
                                    tgt_mask=tgt_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)


class MyTransformer(nn.Module):

    def __init__(self, ntoken_src, ntoken_tgt, d_model=256, nhead=8, dim_ff=2048, nlayers_enc=6, nlayers_dec=6,
                 dropout=0.0):
        super(MyTransformer, self).__init__()

        self.src_pad_idx = 0
        self.sos_token = 3
        self.eos_token = 2

        self.d_model = d_model
        self.embed_src = nn.Embedding(ntoken_src, d_model)
        self.embed_trg = nn.Embedding(ntoken_tgt, d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout)

        self.encoder = Encoder(self.d_model, nhead, dim_ff, nlayers_enc, dropout)
        self.decoder = Decoder(self.d_model, nhead, dim_ff, nlayers_dec, dropout)

        self.linear = nn.Linear(d_model, ntoken_tgt)

    def forward(self, src, tgt, eval=False, max_length=0):

        src_key_padding_mask = self.make_pad_mask(src, self.src_pad_idx)
        tgt_pad_mask = self.make_pad_mask(tgt, self.src_pad_idx)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

        src_key_padding_mask = src_key_padding_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)

        embedded_out_src = self.embed_src(src)
        embedded_out_tgt = self.embed_trg(tgt)

        pos_enc_out_src = self.pos_encoder(embedded_out_src.permute(1, 0, 2))
        pos_enc_out_tgt = self.pos_encoder(embedded_out_tgt.permute(1, 0, 2))

        out = self.encoder(pos_enc_out_src, src_key_padding_mask=src_key_padding_mask)
        # out -> (seq, batch_size, embedding)
        if eval:
            return self.greedy(out, tgt.size(0), src_key_padding_mask, max_length)

        out = self.decoder(pos_enc_out_tgt, out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask,
                           memory_key_padding_mask=src_key_padding_mask)

        out = out.permute(1, 0, 2)
        out = self.linear(out)
        out = out.permute(1, 0, 2)
        return out

    def greedy(self, out, tgt_max_seq_len, src_mask, max_length):
        # out shape (seq, bs, embedd_len)
        bs = out.size(1)
        target_loss = []
        soft = nn.Softmax(dim=1)
        decoder_input = torch.tensor([[self.sos_token] * bs], device=device)
        target_pred = decoder_input.clone()

        for i in range(max_length):
            decoder_input, tgt_mask, tgt_pad_mask = self.prepare_input(decoder_input)
            decoder_out = self.decoder(decoder_input, out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask,
                                       memory_key_padding_mask=src_mask)
            decoder_out = self.linear(decoder_out)
            decoder_out = decoder_out[-1].clone()
            target_loss.append(decoder_out.clone())  # stack matrix to compute loss

            decoder_out = soft(decoder_out)
            decoder_out = torch.unsqueeze(torch.argmax(decoder_out, dim=1), 0)
            target_pred = torch.cat((target_pred, decoder_out), dim=0)
            decoder_input = target_pred.clone()

        target_loss = torch.stack(target_loss, 0)
        return target_pred, target_loss

    def prepare_input(self, decoder_input):
        # decoder_input -> (seq, batch_size)

        tgt_mask = self.generate_square_subsequent_mask(decoder_input.size(0))
        tgt_pad_mask = self.make_pad_mask(decoder_input.permute(1, 0), self.src_pad_idx)

        decoder_input = self.embed_trg(decoder_input)
        decoder_input = self.pos_encoder(decoder_input)

        # decoder_input -> (seq, batch_size, embedding)
        tgt_mask = tgt_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)

        return decoder_input, tgt_mask, tgt_pad_mask

    def make_pad_mask(self, src, pad_idx):
        return (src == pad_idx)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


# Engine
class Engine:

    def __init__(self, model, loss_fn, optimizer, data_collector, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.total_train_loss = []
        self.total_train_acc = []
        self.total_valid_loss = []
        self.total_valid_acc = []
        self.total_batches = 0
        self.data_collector = data_collector

    def train(self, epochs, batch_size, train_data_loader, valid_data_loader):
        total_step = 0
        self.total_train_loss = []
        self.total_valid_loss = []
        self.total_valid_acc = []

        for epoch in range(epochs):

            tmp_train_loss = []
            tmp_train_acc = []

            # with tqdm(train_data_loader, unit="batch") as tepoch:

            for step, batch in enumerate(train_data_loader):
                # for step, batch in enumerate(tepoch):

                # tepoch.set_description(f"Epoch {epoch}")

                self.model.train()
                source = batch[0]
                target = batch[1]

                output = self.model(source, target[:, :-1])

                soft = nn.Softmax(dim=2)
                output_p = soft(output)

                pred = torch.argmax(output_p, dim=2)
                pred = pred.permute(1, 0)
                tr_acc = self.compute_acc(pred, target[:, 1:], 0, 0)

                output = output.permute(1, 2, 0)
                tr_loss = self.loss_fn(output, target[:, 1:])
                tr_loss.backward()

                tmp_train_acc.append(tr_acc)
                tmp_train_loss.append(tr_loss.item())

                if total_step % 10 == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if total_step % 500 == 0:
                    with torch.no_grad():
                        self.model.eval()
                        tr_acc = np.mean(tmp_train_acc)
                        tr_loss = np.mean(tmp_train_loss)
                        val_loss, val_acc = self.compute_valid_eval(epoch, epochs)

                        print(
                            f"Epoch:[{epoch + 1}/{epochs}] Train step {total_step}, Train Loss: {tr_loss}, Valid Loss: {val_loss:}, "
                            f"Train Acc: {tr_acc:.2f}% Valid Acc: {val_acc:.2f}%")
                        tmp_train_acc = []
                        tmp_train_loss = []

                        self.total_train_loss.append([total_step, tr_loss])
                        self.total_train_acc.append([total_step, tr_acc])
                        self.total_valid_loss.append([total_step, val_loss])
                        self.total_valid_acc.append([total_step, val_acc])

                    if self.total_valid_acc:
                        if self.total_valid_acc[-1][1] == 100:
                            print(f"\nModel reached Mean Validation Accuracy:{self.total_valid_acc[-1][1]}")
                            print("Stopping...")
                            return

                total_step += 1


    def compute_valid_eval(self, epoch, epochs):
        val_accuracies = []
        val_losses = []
        correct = 0
        total = 0

        for i, batch in enumerate(valid_data_loader):
            source = batch[0]
            target = batch[1]

            out_pred, out_loss = self.model(source, target, eval=True, max_length=target.size(1) - 1)

            out_pred = out_pred.permute(1, 0)
            out_loss = out_loss.permute(1, 2, 0)

            val_loss = self.loss_fn(out_loss, target[:, 1:])

            val_acc = self.compute_acc(out_pred, target, correct, total)

            val_accuracies.append(val_acc)
            val_losses.append(val_loss.item())

        print("Questions")
        self.translate(source, is_quest=True)
        print("Predicted Answer")
        self.translate(out_pred, is_quest=False)
        return np.mean(val_losses), np.mean(val_accuracies)

    def plot_data(self):
        train_losses = np.array(self.total_train_loss)[:, -1]
        valid_losses = np.array(self.total_valid_loss)[:, -1]
        valid_acc = np.array(self.total_valid_acc)[:, -1]
        train_acc = np.array(self.total_train_acc)[:, -1]
        steps = np.array(self.total_train_loss)[:, 0]

        plt.figure(figsize=(12, 8))
        plt.plot(steps, train_losses, 'b', label="train_loss")
        plt.plot(steps, valid_losses, 'g', label="valid_loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(steps, valid_acc, 'g', label="valid_acc")
        plt.plot(steps, train_acc, 'b', label="train_acc")
        plt.legend()
        plt.show()

    def compute_acc(self, out_pred, target, correct, total):
        correct += (out_pred == target).all(dim=1).sum().item()
        total += target[:, 1:].size(0)
        return 100 * (correct / total)

    def translate(self, batch, is_quest):
        tokens = [0, 1, 2, 3]

        if is_quest:
            quest = []
            for row in range(3):
                for i in batch[row]:
                    if i.item() not in tokens:
                        quest.append(self.data_collector.src_vocab.id_to_string[i.item()])
                print(''.join(quest))
                quest = []
        else:
            answer = []
            for row in range(3):
                for i in batch[row]:
                    if i.item() not in tokens:
                        answer.append(self.data_collector.tgt_vocab.id_to_string[i.item()])
                print(''.join(answer))
                answer = []


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TASK = "numbers__place_value"
    # TASK = "comparison__sort"
    # TASK = "algebra__linear_1d"

    ds = Collector()
    ds.get_dataset("/Users/marcolatella/PycharmProjects/DLL_Assignment_4/numbers__place_value")

    bs = 64
    train_data_loader = DataLoader(
        dataset=ds.train_set, batch_size=bs, shuffle=True)
    ds.train_data_loader = train_data_loader

    valid_data_loader = DataLoader(
        dataset=ds.valid_set, batch_size=bs, shuffle=False)
    ds.valid_data_loader = valid_data_loader

    ntoken_src = len(ds.src_vocab)
    ntoken_tgt = len(ds.tgt_vocab)

    model = MyTransformer(ntoken_src, ntoken_tgt, d_model=256, nhead=8, dim_ff=1024, nlayers_enc=3, nlayers_dec=2)
    model = model.to(device)

    learning_rate = 0.0001

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    engine = Engine(model, loss_fn, optimizer, ds, device)

    epochs = 100
    bs = 64
    engine.train(epochs, bs, ds.train_data_loader, ds.valid_data_loader)
    engine.plot_data()


