CUDA_VISIBLE_DEVICES=0 python train.py --gpu=0 --stage=train --config=configs_moe_step2/mecformer_ctranspath.yaml --fold=0
CUDA_VISIBLE_DEVICES=0 python train.py --gpu=0 --stage=train --config=configs_moe_step2/mecformer_ctranspath.yaml --fold=1
CUDA_VISIBLE_DEVICES=0 python train.py --gpu=0 --stage=train --config=configs_moe_step2/mecformer_ctranspath.yaml --fold=2