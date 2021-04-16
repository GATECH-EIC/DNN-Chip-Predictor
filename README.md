# DNN-Chip Predictor
This is the official implemenatation of [DNN-Chip Predictor: An Analytical Performance Predictor for DNN Accelerators with Various Dataflows and Hardware Architectures [ICASSP'20]](https://arxiv.org/abs/2002.11270)

## Example to predictor the enengy and latency given the operation on [Eyeriss](https://eyeriss.mit.edu/)

### Add your operation to the OPs_list in predictor.py
> + "idx": the index of the operation in the operation set to be processed by predictor.py
> + "type": the type of the operation, should be one from ["Conv", "AvgP", "MaxP", "FC"]
> + "kernel_size": the kernel size of the operation, same with the kernel_size in [PyTorch API](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
> + "stride": the stride of the operation, same with the stride in [PyTorch API](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
> + "padding": the padding size of the operation, same with the padding in [PyTorch API](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
> + "input_H": the heigh of input feature map 
> + "input_W": the width of input feature map 
> + "input_C": the channel of input feature map 
> + "output_E": the heigh of output feature map 
> + "output_F": the width of output feature map 
> + "output_M": the channel of output feature map 

### run the predictor.py
```bash
python predictor.py
```

## Publication

If you use this github repo, please cite:
```
@inproceedings{zhao2020dnn,
author = {Zhao, Yang and Li, Chaojian and Wang, Yue and Xu, Pengfei and Zhang, Yongan and Lin, Yingyan},
year = {2020},
month = {05},
pages = {1593-1597},
booktitle={International Conference on Acoustics, Speech, and Signal Processing},
title = {DNN-Chip Predictor: An Analytical Performance Predictor for DNN Accelerators with Various Dataflows and Hardware Architectures},
doi = {10.1109/ICASSP40776.2020.9053977}
}
```
