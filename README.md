# PRML
模式识别课设-花粉识别

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

### 3. Run the baseline

```python
python train.py
```

## 📁 Project Structure

```python
PRML/
├── Dataset/
├── dataloader.py
├── models.py
├── train.py
└── requirements.txt
```