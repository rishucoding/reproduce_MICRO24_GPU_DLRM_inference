#export LD_PRELOAD=$CONDA_PREFIX/lib/libiomp5.so:$CONDA_PREFIX/lib/libjemalloc.so
export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
PyGenTbl='import sys; rows,tables=sys.argv[1:3]; print("-".join([rows]*int(tables)))'
PyGetCore='import sys; c=int(sys.argv[1]); print(",".join(str(2*i) for i in range(c)))'


# set number of OMP threads, and GPU id: pick between [0,1,2,3,4,5,6,7] based on the availability from nvidia-smi check.
THREADS=1
GPU_ID=2
device="gpu"

data_paths_lst=('/scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/datasets/reuse_high/table_500K.txt' '/scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/datasets/reuse_medium/table_500K.txt' '/scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/datasets/reuse_low/table_500K.txt' '/scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/datasets/random_500K.txt')

hot_paths_lst=('/scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/datasets/pinned_indices/rj_60K_high_hot_indices.txt' '/scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/datasets/pinned_indices/rj_60K_medium_hot_indices.txt' ' /scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/datasets/pinned_indices/rj_60K_low_hot_indices.txt' '/scratch1/rzj5233/problem2/reproduce_MICRO24_GPU_DLRM_inference/datasets/pinned_indices/rj_60K_random_indices.txt')

#data_paths_lst=('/scratch1/rzj5233/problem2/problem2/datasets/random_500K.txt')
#data_paths_lst=('/scratch1/rzj5233/problem2/problem2/datasets/reuse_high/table_500K.txt')

# order is: emb_dim, rows/table, num_tables, pooling_factor
EMBS='128,500000,12,150'
BOT_MLP=256-128-128
TOP_MLP=128-64-1
NUM_BATCH=12 #8 batches are used for warmup and not accounted in the average ET.
BS=2048
cntr=0

for data_path in "${data_paths_lst[@]}"; do
    DATA_GEN_PATH=$data_path
    HOT_FILE=${hot_paths_lst[$cntr]}
    for e in $EMBS; do
        IFS=','; set -- $e; EMB_DIM=$1; EMB_ROW=$2; EMB_TBL=$3; EMB_LS=$4; unset IFS;
        EMB_TBL=$(python -c "$PyGenTbl" "$EMB_ROW" "$EMB_TBL")
        C=$(python -c "$PyGetCore" "$THREADS")
        DATASET_FILE_PATH=$HOT_FILE OMP_NUM_THREADS=$THREADS CUDA_VISIBLE_DEVICES=$GPU_ID taskset -c 0 $CONDA_PREFIX/bin/python load_emb.py --device $device --num-batches $NUM_BATCH --batch-size $BS --lookups-per-sample $EMB_LS --arch-sparse-feature-size $EMB_DIM --arch-embedding-size $EMB_TBL --data-generation=$DATA_GEN_PATH --output-name $cntr
        ((cntr++))
        echo ""
        echo ""
        #echo "Latency of Embedding Bag kernel ****************"
    done
done


#OMP_NUM_THREADS=$THREADS CUDA_VISIBLE_DEVICES=$GPU_ID numactl -C $C -m 0 $CONDA_PREFIX/bin/python  python load_emb.py --device $device --num-batches $NUM_BATCH --batch-size $BS --lookups-per-sample $EMB_LS --arch-sparse-feature-size $EMB_DIM --arch-embedding-size $EMB_TBL --data-generation=$DATA_GEN_PATH


