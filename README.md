# 🧠 Brain Tumor Detection Model

This model detects the type of brain tumor from MRI scans using the VGG16 architecture.

## 💡 Classes
- Glioma
- Meningioma
- No Tumor
- Pituitary

## 🔧 Technologies
- TensorFlow, Keras (VGG16)
- NumPy, PIL, scikit-learn

## 📁 Folder Structure
- `brain_tumor_dataset/` → Your training/testing images
- `models/` → Trained model (`my_model.h5`)
- `requirements` → requirements.txt →  Dependencies
- `src/` → (`main.py`)

## 🚀 To Run:
```bash
pip install -r requirements.txt
python src/main.py
