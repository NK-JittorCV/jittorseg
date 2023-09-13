# JittorSeg

**Coming Soon...**

## Getting Started

### Train
We support single-machine single-gpu, single-machine multi-gpu training, multi-machine training is not supported for the time being. Multi-gpu dependence can be referred to [here](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-2-16-44-distributed/)
```shell
python tools/run_net.py --config-file=path/to/config --task=train

# For example
# Single GPU
python tools/run_net.py --config-file=project/fcn/fcn_r50-d8_512x1024_cityscapes_80k.py --task=train

# Multiple GPUs
mpirun -n 8 python tools/run_net.py --config-file=project/fcn/fcn_r50-d8_512x1024_cityscapes_80k.py --task=train
```

### Val
We provide an evaluation script to evaluate the dataset. If there is not enough CPU memory, you can save CPU memory by setting ```--efficient_val``` to store the evaluation results in a local file.
```shell
python tools/run_net.py --config-file=path/to/config --resume=path/to/ckp --task=val

# For example
python tools/run_net.py --config-file=project/fcn/fcn_r50-d8_512x1024_cityscapes_80k.py --resume=work_dirs/fcn_r50-d8_512x1024_cityscapes_80k/checkpoints/ckpt_80000.pkl --task=val
```

### Test for save result
We provide a test scripts to save the inference results of the data set, which can be saved in the specified location by setting ```--save-dir```.
```shell
python tools/run_net.py --config-file=path/to/config --resume=path/to/ckp --save-dir=path/to/save_dir --task=test

# For example
python tools/run_net.py --config-file=project/fcn/fcn_r50-d8_512x1024_cityscapes_80k.py --resume=work_dirs/fcn_r50-d8_512x1024_cityscapes_80k/checkpoints/ckpt_80000.pkl --save-dir=./ --task=test
```




