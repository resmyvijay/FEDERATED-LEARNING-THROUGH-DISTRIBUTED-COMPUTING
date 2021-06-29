#!/bin/sh python
import sys
import argparse

import numpy as np
import torch as tc
import syft as sf
import grid as gr

parser = argparse.ArgumentParser(
    description="Upload text message data to grid nodes.")

parser.add_argument(
    "-node1",
    "--node1",
    type=str,
    required=True,
    help="Address of the first worker node",
)

parser.add_argument(
    "-node2",
    "--node2",
    type=str,
    required=True,
    help="Address of the second worker node",
)


def main(args):
    # Worker nodes to connect to
    nodes = ["ws://" + args.node1 + "/",
             "ws://" + args.node2 + "/"]

    # Hook into PyTorch from PySyft
    hook = sf.TorchHook(tc)

    # Load workers to be used for learning
    workers = [gr.WebsocketGridClient(hook, nodes[0]),
               gr.WebsocketGridClient(hook, nodes[1])]
    print("Connected to " + args.node1 + " and " + args.node2 + " ...")

    # Load input texts and types into tensors
    types = np.load("data/msgtypes.npy")
    texts = np.load("data/msgtexts.npy")
    texts_data = tc.split(tc.tensor(texts), int(
        len(texts) / len(workers)), dim=0)
    types_data = tc.split(tc.tensor(types), int(
        len(types) / len(workers)), dim=0)

    tag_texts = []
    tag_types = []
    # Tag the message texts and types that willl be stored by the workers
    for i in range(len(workers)):
        tag_texts.append(texts_data[i].tag("#X", "#spam", "#dataset").describe(
            "The message texts for SPAM detection."))
        tag_types.append(types_data[i].tag("#Y", "#spam", "#dataset").describe(
            "The message types for SPAM detection."))

    # Send the data to the workers
    for i in range(len(workers)):
        tensor_txt = tag_texts[i].send(
            workers[i], garbage_collect_data=False)
        tensor_typ = tag_types[i].send(
            workers[i], garbage_collect_data=False)
        print("Worker tensor: ", tensor_txt, tensor_typ)

    # Close connections to all workers
    print("Data upload completed. Closing connections..")
    workers[0].close()
    workers[1].close()
    print("Done.")


if __name__ == '__main__':
    main(parser.parse_args())
