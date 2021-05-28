## Export evaluation statistics
```
CUDA_VISIBLE_DEVICES='0' python tools/train_net.py --config-file ./configs/Cityscapes/context_disentanglement.yaml --num-gpus 1 --eval-only OUTPUT_DIR $OUTPUT_DIR
```

## Run context measuring
```
python context_measurement/entropy_analysis.py $MODEL_DIR (--analyse_mask)
```