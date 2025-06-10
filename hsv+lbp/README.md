## Pollen Classification using HSV + LBP Features

This project implements a traditional image classification pipeline on the POLLEN73S dataset using handcrafted features. It includes preprocessing for size and format normalization, feature extraction (color and texture), and classification using an SVM.

### 🔧 Features
- Handles mixed image formats (JPG, TIFF) and sizes.
- Extracts HSV color histograms and LBP texture descriptors.
- Combines features and uses an SVM classifier for pollen classification.
- Achieves 88% accuracy on the test set over 73 pollen classes.
- Outputs a detailed classification report and auto-generates a Word table for visualization.

### 🚀 How to Run
   ```bash
   python hsv+lbp/main.py
   ```

### 🧰 Requirements
Install the required dependencies with:
```bash
pip install -r requirements.txt
```

#### Minimal `requirements.txt`
```
numpy
opencv-python
scikit-learn
matplotlib
python-docx
```
