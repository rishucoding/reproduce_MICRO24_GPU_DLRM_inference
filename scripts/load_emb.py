## Author: Rishabh Jain
## Parts of doing a basic inference over the embedding bag on GPU
## (1) creating embedding tables on the GPU
## (2) generating the dummy input datasets
## (3) doing embedding inference on the GPU


## --------------------------------------------------------------------
## RJ: putting the imports
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import datetime
import json
import sys
import time

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# data generation
# import dlrm_data_pytorch as dp

# For distributed run
# import extend_distributed as ext_dist
# import mlperf_logger

# numpy
import numpy as np
from numpy import random as ra
# import optim.rwsadagrad as RowWiseSparseAdagrad
import sklearn.metrics

# pytorch
import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
# from torch.utils.tensorboard import SummaryWriter

# mixed-dimension trick
# from tricks.md_embedding_bag import md_solver, PrEmbeddingBag

# quotient-remainder trick
# from tricks.qr_embedding_bag import QREmbeddingBag

## --------------------------------------------------------------------
## RJ: putting all the helper functions here

## Assisting the args parser
def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


## Generating all the embedding tables
def create_emb(m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            n = ln[i]
            # construct embedding operator
            EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
            # initialize embeddings
            # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
            ## RJ: TODO -- is this really necessary to fill these weights?
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            # approach 1
            EE.weight.data = torch.tensor(W, requires_grad=True, dtype=torch.float32)
            v_W_l.append(None)
            emb_l.append(EE)
        return emb_l, v_W_l

def move_tables_to_gpu(emb_l):
    cuda0 = torch.device("cuda:0")
    g_emb = []
    with torch.cuda.device(cuda0):
        ## move the tables to GPU
        for x in range(len(emb_l)):
            g_emb.append(emb_l[x].to(cuda0))
    return g_emb




## Doing the inference where embedding lookups happens, for a BATCH
## Two versions of this function such that in one
## we don't count the time for data movement (only kernel ET)
def apply_emb(lS_o, lS_i, g_emb):
        cuda0 = torch.device("cuda:0")

        ly = []
        print()
        start = time.perf_counter()
        with torch.cuda.device(cuda0):
            ## move the datasets to GPU
            for x in range(len(g_emb)):
                g_indices = (lS_i[x].to(cuda0))
                g_offsets = (lS_o[x].to(cuda0))
                torch.cuda.synchronize()
                per_sample_weights = None

                g_E = g_emb[x]
                ln_start = time.perf_counter()
                V = g_E(
                    g_indices,
                    g_offsets,
                    per_sample_weights=per_sample_weights,
                )
                torch.cuda.synchronize()
                ln_end = time.perf_counter()
                print(ln_end - ln_start)

                ly.append(V)
            print()
        end = time.perf_counter()
        return end-start, ly

# purely measure kernel execution time
# try to match the time shown by nsys profiler
def apply_emb2(lS_o, lS_i, g_emb):
        cuda0 = torch.device("cuda:0")

        g_indices = []
        g_offsets = []
        ly = []
        kernel_et = 0

        with torch.cuda.device(cuda0):
            ## move the datasets(indices/offsets array) to GPU in one Shot
            for x in range(len(g_emb)):
                #print(x)
                g_indices.append((lS_i[x].to(cuda0)))
                g_offsets.append((lS_o[x].to(cuda0)))
                torch.cuda.synchronize()
                per_sample_weights = None

            ## process the GatherReduce compute of all tables in one shot
            start = time.perf_counter()
            for x in range(len(g_emb)):
                g_E = g_emb[x]

                V = g_E(
                    g_indices[x],
                    g_offsets[x],
                    per_sample_weights=per_sample_weights,
                )
                ly.append(V) # collect the outputs to be used in the later stages
            torch.cuda.synchronize()
            end = time.perf_counter()
            kernel_et += end-start

        return kernel_et, ly



## --------------------------------------------------------------------
## RJ: generating the inputs
# uniform ditribution (input data), n is the mini batch size -- that many inputs needed in a batch
# this function is called for each batch
def generate_uniform_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    #num_indices_per_lookup_fixed,
    #length,
):
    # dense feature <- for BMLP
    ##TODO this throws error: "RuntimeError: Could not infer dtype of numpy.float32"
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32), dtype=torch.float32)
    #Xt = (ra.rand(n, m_den).astype(np.float32))
    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    #RJ: for each table
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        #RJ: goto each sample
        for _ in range(n):
            #pooling factor for each sample
            sparse_group_size = np.int64(num_indices_per_lookup)
            # sparse indices to be used per embedding
            r = ra.random(sparse_group_size)
            #sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
            sparse_group = (np.round(r * (size - 1)).astype(np.int64))
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int32(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets, dtype=torch.int64))
        lS_emb_indices.append(torch.tensor(lS_batch_indices, dtype=torch.int64))

    return (Xt, lS_emb_offsets, lS_emb_indices)


## RJ: reading  the trace files to generate the datasets
## Mechansim: for a given trace file, we sequentially read the indices to assign for a
## batch for each table, and for all batches. We do a circular read if we reach the end
## of the trace file.

def open_gen(name, rows):
    with open(name) as f:
        idx = list(filter(lambda x: x < rows, map(int, f.readlines())))
    while True:
        for x in idx:
            yield x

dataset_gen = None

def get_gen(fname, rows):
    global dataset_gen
    if dataset_gen is None:
        #_, fname, rows = dgen.split(',')

        # '/home/cc/dataset/items_in_buy.txt'
        # '/home/cc/Downloads/dlrm_datasets/table_fbgemm_t856_bs65536_0/table_0.txt'
        dataset_gen = open_gen(fname, int(rows))
    return dataset_gen


def trace_read_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    fname,
    #num_indices_per_lookup_fixed,
    #length,
):
    # dense feature <- for BMLP
    ##TODO this throws error: "RuntimeError: Could not infer dtype of numpy.float32"
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32), dtype=torch.float32)
    #Xt = (ra.rand(n, m_den).astype(np.float32))
    #print(">>>>>>>>>> From RJ: passing the #rows: " + str(ln_emb[0]))
    cur_gen = get_gen(fname, ln_emb[0])

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    #RJ: for each table
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        #RJ: goto each sample
        for _ in range(n):
            #pooling factor for each sample
            sparse_group_size = np.int64(num_indices_per_lookup)
            # sparse indices to be used per embedding
            r = ra.random(sparse_group_size)
            #sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
            # reset sparse_group_size in case some index duplicates were removed
            #sparse_group_size = np.int32(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += [x for _, x in zip(range(sparse_group_size), cur_gen)]
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets, dtype=torch.int64))
        lS_emb_indices.append(torch.tensor(lS_batch_indices, dtype=torch.int64))

    return (Xt, lS_emb_offsets, lS_emb_indices)


## Rishabh: helper function
def print_model_config(device, nbatches, n, table_config, emb_dim, lookups_per_sample, fname):
    print("Device: " + device)
    print("Hotness: " + fname)
    print("Num batches: " + str(nbatches))
    print("Batch Size: " + str(n))
    print("Table config: " + str(table_config))
    print("Embedding Dimension " + str(emb_dim))
    print("Lookups per sample: " + str(lookups_per_sample))



## --------------------------------------------------------------------


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="DLRM Inference on CPU and GPUs"
    )

    # emb related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=128)
    parser.add_argument("--arch-embedding-size", type=dash_separated_ints, default="4-3-2")

    # MLP related parameters
    #parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    #parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")

    # execution and dataset related parameters
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--data-generation", type=str, default="/scratch/rzj5233/problem2/problem2/datasets/reuse_high/table_500K.txt")
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--output-name", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lookups-per-sample", type=int, default=150)

    global args
    args = parser.parse_args()

    device = args.device
    nbatches = args.num_batches
    warmups = 0

    numpy_rand_seed = 512
    np.random.seed(numpy_rand_seed)
    #ln_emb = [500000]*12;
    ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
    ln_emb = np.asarray(ln_emb, dtype=np.int32)
    m_spa = args.arch_sparse_feature_size #embedding dim
    n = args.batch_size # batch size
    fname = args.data_generation
    num_indices_per_lookup = args.lookups_per_sample # pooling factor or lookups per sample
    ln_bot = np.fromstring('256-128-' + str(m_spa), dtype=int, sep="-")
    lS_o = []
    lS_i = []

    print_model_config(device, nbatches, n, args.arch_embedding_size, m_spa, num_indices_per_lookup, fname)

    start = time.perf_counter()
    for j in range(0, nbatches):
        #Xt, lS_emb_offsets, lS_emb_indices = generate_uniform_input_batch(ln_bot[0], ln_emb, n, num_indices_per_lookup)
        Xt, lS_emb_offsets, lS_emb_indices = trace_read_input_batch(ln_bot[0], ln_emb, n, num_indices_per_lookup, fname)
        lS_o.append(lS_emb_offsets)
        lS_i.append(lS_emb_indices)

    #emb_l, w_list = create_emb(m_spa, ln_emb)

    end = time.perf_counter()
    print('Time elapsed(s) in model and data gen: {:10.6f}'.format(end-start))

    # This includes model loading and running inference.

    tot_emb_time = 0
    model_loading_time = 0
    time_each_batch = []
    output_tensor = []
    if(device == "gpu"):
        if torch.cuda.is_available():
            #g_emb = move_tables_to_gpu(emb_l)
            #g_emb = torch.load('/i3c/hpcl/rzj5233/hetsys/dlrm/problem2/problem2/hacking_param/train/compute/pt/rough/problem2/gpu_setup/save_load_models/test.pt')
            lv_time_start = time.perf_counter()
            g_emb = torch.load('/scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/scripts/test_500K_12.pt')
            lv_time_end = time.perf_counter()
            model_loading_time = lv_time_end - lv_time_start
            print('Model loading time(s): {:10.6f}'.format(model_loading_time))

            #g_emb = move_tables_to_gpu(g_emb)
            for x in range(warmups + nbatches):
                l_emb_time, ly = apply_emb2(lS_o[x-warmups], lS_i[x-warmups], g_emb)
                output_tensor.append(ly)
                #count the time one later since 1st batch takes very long
                if(x >= warmups):
                    tot_emb_time += l_emb_time
                    time_each_batch.append(l_emb_time)
        else:
            print("Seems like CUDA is not available!!")
    else:
        ## RJ: TODO -- adding the support for the CPU, a project on heterogeneous computing.
        print("Please set device as gpu!")

    torch.save(output_tensor, '/scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/scripts/' + str(args.output_name) + '.pt')
    print('Time elapsed(s): {:10.6f}'.format(tot_emb_time))
    # print('Time elapsed(s) on avg batch: {:10.6f}'.format(sum(time_each_batch[1:])/len(time_each_batch[1:])))
    # print('Time each batch: ')
    print(time_each_batch)
    if(nbatches > 8):
        avg_batch_lat = sum(time_each_batch[8:])/ len(time_each_batch[8:])
        print('Average Batch Latency(s):{:10.6f}'.format(avg_batch_lat))
    else:
        print('Nothing to average: num_batches <=8')


if __name__ == "__main__":
    run()
