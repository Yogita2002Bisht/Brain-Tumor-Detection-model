# Brain Tumor Detection Model
This repository contains my work on brain tumor detection using deep learning (VGG16).
The current model classifies MRI scans into multiple categories of brain tumors and non-tumor cases.

I aim to continue improving this project by moving towards segmentation and 3D visualization for more clinical relevance, but this version demonstrates a complete and working classification pip

 **Key Contributions**
-Built a brain tumor classification model using transfer learning (VGG16).
-Achieved high accuracy on test MRI scans with strong generalization.
-Designed a modular pipeline for preprocessing, training, and evaluation.
-Implemented visual analysis tools (confusion matrix, prediction plots) for model interpretability.

## Classes
- Glioma
- Meningioma
- No Tumor
- Pituitary

**Current Work (Stage 1)**
-Dataset: Brain MRI images.
-Model: Transfer learning with VGG16.
-Preprocessing: Normalization, resizing, and augmentation.
-Training: Implemented using TensorFlow/Keras.
-Output: Binary classification (Tumor / No Tumor).

**Results**
Achieved strong accuracy on test images
Predictions visualized using confusion matrix & plots

**Future Direction**

Next steps include:

-Adding tumor boundary segmentation for precise localization
-Exploring 3D visualization of MRI scans for more intuitive interpretation

**Vision**
This project reflects my interest in AI for healthcare and social good.
While the current model focuses on detection, the broader goal is to build trustworthy, explainable, and clinically useful tools for supporting doctors and improving patient care.

## Technologies
-Deep Learning: TensorFlow, Keras (VGG16)
-Data Processing: NumPy, PIL, scikit-learn
-Visualization: Matplotlib, Seaborn

## Folder Structure
- `brain_tumor_dataset/` → Your training/testing images
- `models/` → Trained model (`my_model.h5`)
- `requirements` → requirements.txt →  Dependencies
- `src/` → (`main.py`)

## To Run:
```bash
pip install -r requirements.txt
python src/main.py
