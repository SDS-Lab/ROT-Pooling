# ROT-Pooling
* Regularized Optimal Transport Layers for Generalized Global Pooling Operations [https://ieeexplore.ieee.org/document/10247589]. 
* The work is an extension of "Revisiting Global Pooling through the Lens of Optimal Transport" [https://arxiv.org/pdf/2201.09191.pdf].

## Dependencies

* [PyTorch] - Version: 1.10.0
* [PyTorch Geometric] - Version: 2.0.3

## Training & Evaluation


* attention_mil

```
python attention_mil.py --DS 'datasets/messidor' --pooling_layer 'uot_pooling' --f_method 'sinkhorn' --num 4 
```


* adgcl

```
python adgcl.py --DS 'IMDB-BINARY' --pooling_layer 'rot_pooling' --f_method 'badmm-e' --num 4
```

* ddi

```
python ddi_gin.py --DS 'fears' --pooling_layer 'uot_pooling' --f_method 'badmm-e' --num 4
```

* resnet-imagenet


The setting of parameters refer to the github link: https://github.com/pytorch/examples/tree/main/imagenet

```
python resnet_imagenet.py --f_method 'badmm-e' --num 4
```

## Parameters


```DS``` is the dataset.

```pooling_layer``` is the pooling layer chosen for the backbone, including add_pooling, mean_pooling, max_pooling, deepset, 
mix_pooling, gated_pooling, set_set, attention_pooling, gated_attention_pooling, dynamic_pooling, GeneralizedNormPooling,
SAGPooling, ASAPooling, OTK, SWE, WEGL, uot_pooling, rotpooling. Uot_pooling corresponds to "ROTP(a_0=0)" and rot_pooling corresponds to 
"ROTP(learned a_0)" in the paper.

```f_method``` could be ```badmm-e, badmm-q, sinkhorn``` 

```num``` corresponds to K-step feed-forward computation. The default value is 4.

## Citation

If our work can help you, please cite it
```
@ARTICLE{10247589,
  author={Xu, Hongteng and Cheng, Minjie},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Regularized Optimal Transport Layers for Generalized Global Pooling Operations}, 
  year={2023},
  volume={},
  number={},
  pages={1-18},
  doi={10.1109/TPAMI.2023.3314661}}
```

