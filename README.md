# Res-Reg
## Installation

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget

## Code Overview

#### Main Files

- `main.py`: main training and evaluation script
- `data_split.py`: Split the dataset into train, val, and test 
- `data_prepare.py`: create data file `data.csv` with balanced val/test set

## Getting Started

#### Split dataset

```bash
python data_split.py
```

#### Create data file 'data.csv'

```bash
python data_prepare.py
```

#### Train the Res-Reg
```bash
python main.py
```
