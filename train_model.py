#!/bin/sh python
import math
import numpy as np
import torch as tc

import argparse

import syft as sf
import grid as gr

from torch import nn
from GRU_RNN import GRU

parser = argparse.ArgumentParser(
    description="Perform training and bechmark of spam dataset.")

parser.add_argument(
    "-gw",
    "--gateway",
    type=str,
    required=True,
    help="Address of the grid gateway",
)


def main(args):
    hook = sf.TorchHook(tc)

    tr_grid = gr.GridNetwork("http://" + args.gateway)
    texts = tr_grid.search("#X", "#spam", "#dataset")
    types = tr_grid.search("#Y", "#spam", "#dataset")
    print("Found datasets:")
    print(texts)
    print(types)

    # Model parameters
    HIDDEN_DIM = 10
    EMBEDDING_DIM = 50
    DROPOUT = 0.2
    EPOCHS = 15
    BATCH_SIZE = 32
    HIDDEN_DIM = 10
    LR = 0.1
    VOCAB_SIZE = 0
    for texts_comp in texts:
        VOCAB_SIZE = max(VOCAB_SIZE, int(texts_comp[0].max().get()))
    VOCAB_SIZE += 1

    # Initiating the model
    model = GRU(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM,
                embedding_dim=EMBEDDING_DIM, dropout=DROPOUT)

    # Defining the loss and optimizer
    criterion = nn.BCELoss()
    optimizer = tc.optim.SGD(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        dataset_size = sum([len(texts[i][0]) for i in range(len(texts))])
        print("Beginning training ...")
        model.train()

        for i in range(len(texts)):
            loss_acc = 0
            nr_batches = math.ceil(len(texts[i][0]) / BATCH_SIZE)
            for batch_idx in range(nr_batches):
                # Extract the batch for training and target
                texts_batch = texts[i][0][BATCH_SIZE *
                                          batch_idx: BATCH_SIZE * (batch_idx + 1), :]
                types_batch = types[i][0][BATCH_SIZE *
                                          batch_idx: BATCH_SIZE * (batch_idx + 1)]

                # Send the model to the worker
                worker = texts_batch.location
                model.send(worker)
                h = tc.Tensor(tc.zeros((BATCH_SIZE, HIDDEN_DIM))).send(worker)

                optimizer.zero_grad()
                pred, _ = model(texts_batch, h)
                loss = criterion(pred.squeeze(), types_batch.float())
                loss.backward()
                optimizer.step()
                model.get()

                # Accumulate the loss
                loss_acc += loss.get().item()

            print('Epoch: {} [{}/{} ({:.0f}%)]\tTraining loss: {:.6f}'.format(
                epoch, i * texts[i][0].shape[0], dataset_size,
                100. * (i*texts[i][0].shape[0]) / dataset_size, loss_acc))


if __name__ == '__main__':
    main(parser.parse_args())
