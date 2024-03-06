# Copyright (C) 2024 Cyprien Quéméneur
# FedPylot is released under the GPL-3.0 license, please refer to the LICENSE file in the root directory of the program.
# For the full copyright notices, please refer to the NOTICE file in the root directory of the program.

import argparse
import os
import shutil
import sys
sys.path.append('yolov7')

from mpi4py import MPI
import pandas as pd
import yaml

from node import Client, Node, Server


def init_node(rank: int, server_opt: str, server_lr: float, tau: float, beta: float) -> Node:
    """Initialize a node (client or server) based on its rank."""
    if server_opt not in ['fedavg', 'fedavgm', 'fedadagrad', 'fedadam']:
        raise ValueError(f'Server optimizer {server_opt} unavailable, must be fedavg, fedavgm, fedadagrad, or fedadam.')
    return Server(server_opt, server_lr, tau, beta) if rank == 0 else Client(rank)


def share_public_keys(node: Node) -> None:
    """Clients receive the server's public key while the server receives each client's public key."""
    key_rank_pairs = comm.gather((node.rank, node.public_key), root=0)
    if node.rank == 0:
        key_rank_pairs = {r: cpk for r, cpk in key_rank_pairs}
        key_rank_pairs.pop(0)
        node.clients_public_keys = key_rank_pairs
    public_key = comm.bcast(node.public_key, root=0)
    if node.rank != 0:
        node.server_public_key = public_key


def share_symmetric_key(node: Node) -> None:
    """The symmetric key is generated by the server and shared among the clients using asymmetric encryption."""
    if node.rank == 0:
        node.generate_symmetric_key()
        sk = [None] + node.get_symmetric_key()
    else:
        sk = None
    sk = comm.scatter(sk, root=0)
    if node.rank != 0:
        node.symmetric_key = sk


def initial_broadcast(node: Node, pretrained_weights: str, data: str, cfg: str, hyp: str, imgsz: int) -> None:
    """The central server initializes the checkpoint from a pretrained weights file and shares it with the clients."""
    if node.rank == 0:
        node.initialize_model(pretrained_weights)
        encrypted_data = [None] + node.get_weights(metadata=True)
        node.post_init_update(data=data, cfg=cfg, hyp=hyp, imgsz=imgsz)
    else:
        encrypted_data = None
    encrypted_data = comm.scatter(encrypted_data, root=0)
    if node.rank != 0:
        node.set_weights(encrypted_data, metadata=True)


def federated_loop(node: Node, nrounds: int, epochs: int, saving_path: str, architecture: str, pretrained_weights: str,
                   data: str, bsz_train: int, bsz_val: int, imgsz: int, conf_thres: float, iou_thres: float, cfg: str,
                   hyp: str, workers: int) -> None:
    """Orchestrate the federated learning experiment."""
    for kround in range(nrounds):
        # At the beginning of a round, generate and share a new symmetric key
        share_symmetric_key(node)
        # If it is the first round, the central server sends the initial checkpoint to the clients
        if kround == 0:
            initial_broadcast(node, pretrained_weights, data, cfg, hyp, imgsz)
        # Client level computation (local training)
        if node.rank != 0:
            node.train(nrounds, kround, epochs, architecture, data, bsz_train, imgsz, cfg, hyp, workers, saving_path)
            sd_encrypted = node.get_update()
        else:
            sd_encrypted = None
        # Updates are gathered by the central server
        sd_encrypted = comm.gather(sd_encrypted, root=0)
        # Server level computation (server optimization, re-parameterization, and evaluation on the validation set)
        if node.rank == 0:
            sd_encrypted.pop(0)
            node.aggregate(sd_encrypted)
            node.reparameterize(architecture)
            node.test(kround, saving_path, data, bsz_val, imgsz, conf_thres, iou_thres)
            sd_encrypted = [None] + node.get_weights(metadata=False)
        else:
            sd_encrypted = None
        # New weights are shared with the clients
        sd_encrypted = comm.scatter(sd_encrypted, root=0)
        if node.rank != 0:
            node.set_weights(sd_encrypted, metadata=False)


def gather_analytics(saving_path: str, node: Node) -> None:
    """Gather local analytics from the client nodes' local storage back to the server's local storage."""
    os.makedirs(f'{saving_path}/run/local-analytics/')
    if node.rank != 0:
        df_lr = pd.read_csv(f'{saving_path}/run/train-client{node.rank}/optim_params.csv')
        df_loss = pd.read_csv(f'{saving_path}/run/train-client{node.rank}/training_losses.csv')
        with open(f'{saving_path}/run/train-client{node.rank}/opt.yaml') as f:
            save_yaml = yaml.load(f, Loader=yaml.SafeLoader)
        local_analytics = (df_lr, df_loss, save_yaml)
    else:
        local_analytics = None, None, None
    local_analytics = comm.gather(local_analytics, root=0)
    if node.rank == 0:
        for i, (df_lr, df_loss, save_yaml) in enumerate(local_analytics):
            if i != 0:
                df_lr.to_csv(f'{saving_path}/run/local-analytics/optim_params_{i}.csv')
                df_loss.to_csv(f'{saving_path}/run/local-analytics/training_losses_{i}.csv')
                with open(f'{saving_path}/run/local-analytics/opt_{i}.yaml', 'w') as f:
                    yaml.dump(save_yaml, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrounds', type=int, default=30, help='number of communication rounds')
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs executed per communication round')
    parser.add_argument('--server-opt', type=str, default='fedavg', help='aggregation algorithm/server-side optimizer')
    parser.add_argument('--server-lr', type=float, default=1., help='server learning rate')
    parser.add_argument('--tau', type=float, default=1e-3, help='server adaptivity level with FedAdam and FedAdagrad')
    parser.add_argument('--beta', type=float, default=0.1, help='server momentum with FedAvgM')
    parser.add_argument('--architecture', type=str, default='yolov7', help='model architecture')
    parser.add_argument('--weights', type=str, help='path to pretrained weights')
    parser.add_argument('--data', type=str, help='*.data path')
    parser.add_argument('--bsz-train', type=int, default=32, help='batch size used for training')
    parser.add_argument('--bsz-val', type=int, default=32, help='batch size used for evaluation')
    parser.add_argument('--img', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--cfg', type=str, default='yolov7/cfg/training/yolov7.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, help='hyperparameters path, also decides client-side optimizer')
    parser.add_argument('--workers', type=int, default=8, help='number of workers to use during training')
    args = parser.parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    node = init_node(rank, args.server_opt, args.server_lr, args.tau, args.beta)
    node.get_device_info()

    # Save the number of training examples held by the clients to perform weighted average aggregation of the updates
    with open(args.data) as f:
        if node.rank != 0:
            img_path = os.path.join(yaml.load(f, Loader=yaml.SafeLoader)['train'], f'client{node.rank}', 'images')
            node.nsamples = len(os.listdir(img_path))

    # The clients exchange their public keys with the central server and vice-versa
    share_public_keys(node)

    # Create saving folder
    saving_path = 'experiments'
    os.makedirs(saving_path)
    os.makedirs(saving_path + '/weights/')
    os.makedirs(saving_path + '/run/')

    # Save config, cfg, hyp and data files
    if node.rank == 0:
        with open(saving_path + '/config.txt', 'w') as f:
            f.write(f'nrounds: {args.nrounds}\n')
            f.write(f'epochs: {args.epochs}\n')
            f.write(f'server opt: {args.server_opt}\n')
            f.write(f'server learning rate: {args.server_lr}\n')
            if args.server_opt == 'fedavgm':
                f.write(f'fedavgm - beta: {args.beta}\n')
            if args.server_opt == 'fedadagrad':
                f.write(f'fedadagrad - tau: {args.tau}\n')
            if args.server_opt == 'fedadam':
                f.write(f'fedadam - tau: {args.tau}\n')
                f.write(f'fedadam - beta1: {0.9}\n')
                f.write(f'fedadam - beta2: {0.99}\n')
            f.write(f'architecture: {args.architecture}\n')
            f.write(f'weights: {args.weights}\n')
            f.write(f'data: {args.data}\n')
            f.write(f'batch size (train): {args.bsz_train}\n')
            f.write(f'batch size (eval): {args.bsz_val}\n')
            f.write(f'img: {args.img}\n')
            f.write(f'conf: {args.conf}\n')
            f.write(f'iou: {args.iou}\n')
            f.write(f'cfg: {args.cfg}\n')
            f.write(f'hyp: {args.hyp}\n')
            f.write(f'workers: {args.workers}\n')
        shutil.copy(args.cfg, saving_path)
        shutil.copy(args.hyp, saving_path)
        shutil.copy(args.data, saving_path)

    # Launch federated learning experiment
    federated_loop(
        node=node,
        nrounds=args.nrounds,
        epochs=args.epochs,
        saving_path=saving_path,
        architecture=args.architecture,
        pretrained_weights=args.weights,
        data=args.data,
        bsz_train=args.bsz_train,
        bsz_val=args.bsz_val,
        imgsz=args.img,
        conf_thres=args.conf,
        iou_thres=args.iou,
        cfg=args.cfg,
        hyp=args.hyp,
        workers=args.workers
    )

    # Gather clients' local analytics back to the orchestrating node in order to back up the files
    gather_analytics(saving_path, node)
