# PRML
模式识别课设-48-花粉识别

## 🚀Getting Start

### 1. Clone the repository

```python
git clone https://github.com/HaotianYang03/PRML
cd PRML
```

### 2. Create environment

```python
conda create -n PRML python==3.10
conda activate PRML
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Train

#### 3.1 Run the baseline

```python
python train.py --model ViT --lr 0.0001 --epochs 100 --t_max 100 --eta_min 1e-6
```

#### 3.2 Run ViT_B_16

```python
python train.py --model ViT_B_16 --lr 0.0001 --epochs 200 --t_max 100 --eta_min 1e-6
```

#### 3.3 Run ViT_H_14

```python
python train.py --model ViT_H_14 --lr 0.0001 --epochs 200 --t_max 100 --eta_min 1e-6
```

## 📁 Project Structure

```python
PRML/
├── Dataset/
│      ├──acrocomia_aculeta/
│      ├──anadenanthera_colubrina/
│      └──...
├── dataloader.py
├── models.py
├── train.py
└── requirements.txt
```