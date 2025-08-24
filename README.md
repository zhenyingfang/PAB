# PAB

official code for PAB

## Preprocess

### Environment

- Our code is built upon the codebase from [OpenTAD](https://github.com/sming256/OpenTAD), and we would like to express our gratitude for their outstanding work.

### Data Process

Please follow the data processing of OpenTAD to obtain the ActivityNet-1.3 dataset.

Add the obtained activitynet-1.3 dataset to the data directory. The directory structure is as follows:

── data \
   ├── ablation_annos \
   └── activitynet-1.3

## Train PAB

- Run the `train.sh` script for training

## Training logs and checkpoints

- [BaiduDisk](https://pan.baidu.com/s/1X7yw72_61JgzJq5vUCXyhQ?pwd=vghg) (code: vghg)

## Results

| <center>Split</center> | 0.5  | 0.75  | 0.95  | Avg.  |
| ---------------------- | ---- | ---- | ---- | ---- |
| 10% fully supervised | 38.6 | 25.5 | 5.5 | 25.2 |
| 10% fully + 10% video label + 10% point label  | 41.4 | 27.7 | 6.2 | 27.1 |
