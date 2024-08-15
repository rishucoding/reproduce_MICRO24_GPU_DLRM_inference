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


## Generating all the embedding tables
"""
Aim: obtain a list of tables (embedding stage)
* Generate each table using the associated row/colm dimension.
* Populate random values in the table to create a dummy model.
*(We restrict not populating zeroes since some compiler optimization could bypass the gather reduce operations).
"""
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
## TODO: let's have two versions of this function such that in one
## we don't count the time for data movement (only kernel ET)
def apply_emb(lS_o, lS_i, g_emb):
        cuda0 = torch.device("cuda:0")

        #g_indices = []
        #g_offsets = []
        ly = []
        start = time.perf_counter()
        with torch.cuda.device(cuda0):
            ## move the datasets to GPU
            for x in range(len(g_emb)):
                #g_indices.append(lS_i[x].to(cuda0))
                #g_offsets.append(lS_o[x].to(cuda0))
                g_indices = (lS_i[x].to(cuda0))
                g_offsets = (lS_o[x].to(cuda0))
                torch.cuda.synchronize()
                per_sample_weights = None

                g_E = g_emb[x]
                V = g_E(
                    g_indices,
                    g_offsets,
                    per_sample_weights=per_sample_weights,
                )
                torch.cuda.synchronize()
                ly.append(V)
        end = time.perf_counter()
        return end-start, ly

# purely measure kernel execution time
# try to match the time shown by nsys profiler
def apply_emb2(lS_o, lS_i, g_emb):
        cuda0 = torch.device("cuda:0")

        #g_indices = []
        #g_offsets = []
        ly = []
        kernel_et = 0

        with torch.cuda.device(cuda0):
            ## move the datasets to GPU
            for x in range(len(g_emb)):
                g_indices = (lS_i[x].to(cuda0))
                g_offsets = (lS_o[x].to(cuda0))
                torch.cuda.synchronize()
                per_sample_weights = None

            start = time.perf_counter()
            for x in range(len(g_emb)):
                g_E = g_emb[x]

                V = g_E(
                    g_indices,
                    g_offsets,
                    per_sample_weights=per_sample_weights,
                )
                #ly.append(V)
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

## --------------------------------------------------------------------

device = "gpu"
nbatches = 8
#RJ TODO: how to properly use warmups such that I don't use the indices of the actual batches??
warmups = 0
lS_o = []
lS_i = []


numpy_rand_seed = 512
np.random.seed(numpy_rand_seed)
#ln_emb = [50000, 100000, 200000, 400000] # list tables with their row sizes
#ln_emb = [20000000]
ln_emb = [500000]*12;
#ln_emb = [200000]*12;
#ln_emb = [500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000, 500000]
ln_emb = np.asarray(ln_emb, dtype=np.int32)
m_spa = 128 #embedding dim
n = 2048 # batch size
num_indices_per_lookup = 150 # pooling factor or lookups per sample
ln_bot = np.fromstring('256-128-' + str(m_spa), dtype=int, sep="-")
#fname = '/i3c/hpcl/rzj5233/hetsys/dlrm/problem2/problem2/hacking_param/train/compute/pt/rough/problem2/datasets/one_item.txt'
fname = '/i3c/hpcl/rzj5233/hetsys/dlrm/problem2/problem2/hacking_param/train/compute/pt/rough/problem2/datasets/high_500K.txt'

start = time.perf_counter()

emb_l, w_list = create_emb(m_spa, ln_emb)

end = time.perf_counter()
print('Time elapsed(s) in model and data gen: {:10.6f}'.format(end-start))

tot_emb_time = 0
time_each_batch = []
if(device == "gpu"):
    if torch.cuda.is_available():
        g_emb = move_tables_to_gpu(emb_l)
        torch.save(g_emb, '/scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/scripts/test_500K_12.pt')

    else:
        print("Seems like CUDA is not available!!")
else:
    print("Please set device as gpu!")

print('Time elapsed(s): {:10.6f}'.format(tot_emb_time))
