for ds in IMDB-BINARY
do
  echo $ds
  for pooling in uot_pooling rot_pooling
  do
    for method in badmm-e badmm-q sinkhorn
    do
        echo $pooling
        for seed in 0 1 2 3 4
        do
          CUDA_VISIBLE_DEVICES=0 python adgcl.py  --num 4 --DS $ds --pooling_layer $pooling --f_method $method --seed $seed
        done
    done
  done
done


