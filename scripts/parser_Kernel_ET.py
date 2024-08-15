## From Rishabh Jain
## Aim: to find the average Kernel ET
## Total samples: num_batches * num_tables = 12 * 12 = 144

from collections import Counter
import re
import torch
import gc
import random
import glob

filename = '/scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/scripts/log'
et_list = []
with open(filename, "r") as file:
  for line in file:
    # Process each line here
    et_list.append(float(line.strip()))  # Remove trailing newline character

num_tables = 12
num_batches = 12

# remove the cold miss: 1st kernel
for x in range(96):
    start = x * num_batches * num_tables
    end = (x+1) * num_batches * num_tables
    avg_list = et_list[start+1:end]
    #print(len(avg_list))
    print('{:10.6f}'.format(float(sum(avg_list)/len(avg_list))))
    #print(float(sum(avg_list)/len(avg_list)))
