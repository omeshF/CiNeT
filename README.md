# Classify in Network Transformation ( CiNeT )

This project provides **dynamic CNN classifiers** with **GPU/RAM monitoring** in both **TensorFlow/Keras** and **PyTorch**.  
The system automatically detects classes, splits data, optimizes batch size, and provides detailed evaluation with memory usage tracking.

---

## 🚀 Features
- Auto-detect classes from `data_` folders  
- Adapts for **binary** and **multi-class** tasks  
- Real-time **GPU & RAM monitoring**  
- Automatic **train/val/test split**  
- Confusion matrix, classification report, and metrics  
- Memory usage visualization + batch size optimization  

---

## 📦 Installation

### TensorFlow Version
```bash
pip install tensorflow opencv-python seaborn matplotlib scikit-learn psutil
```

### PyTorch Version
```bash
pip install torch torchvision tqdm seaborn opencv-python matplotlib scikit-learn psutil
```

### Optional (GPU monitoring)
```bash
pip install pynvml
```

---

## 📂 Data Structure
```
/home/ubuntu/Images/
├── data_class1/
│   ├── image1.jpg
│   └── ...
├── data_class2/
│   ├── image1.jpg
│   └── ...
└── data_classN/
    ├── image1.jpg
    └── ...
```

---

## ⚙️ Usage

### TensorFlow/Keras
```python
classifier = DynamicCNN(num_classes=None, data_dir="/home/ubuntu/Images/")
classifier.check_system_resources()
classifier.prepare_data()
classifier.build_model()
classifier.create_data_generators()
classifier.train(epochs=50)
classifier.evaluate_model()
classifier.plot_memory_usage()
```

### PyTorch
```python
classifier = DynamicCNN(num_classes=None, data_dir="/home/ubuntu/Images/")
classifier.check_system_resources()
classifier.prepare_data()
classifier.build_model()
classifier.create_loaders()
classifier.train(epochs=50)
classifier.evaluate()
classifier.memory_monitor.plot_usage()
```

---

## 🧠 Model Architectures
- **TensorFlow/Keras**: 3 Conv blocks + BatchNorm + Dropout  
- **PyTorch**: 4 Conv blocks + BatchNorm + Dropout  

---

## 📊 Evaluation
- Accuracy/loss plots  
- Confusion matrix & classification report  
- Memory usage tracking (RAM + GPU)  
- Batch size optimization  

---

## 🔄 Data Conversion
Generate images from network traffic:  
- **NeT2I** → [GitHub](https://github.com/omeshF/NeT2I) | [PyPI](https://pypi.org/project/net2i/)  

Convert images back to traffic:  
- **I2NeT** → [GitHub](https://github.com/omeshF/I2NeT) | [PyPI](https://pypi.org/project/i2net/)  

---

## 📖 Citation
```bibtex
@article{202508.1085,
  doi = {10.20944/preprints202508.1085.v2},
  url = {https://doi.org/10.20944/preprints202508.1085.v2},
  year = 2025,
  month = {August},
  publisher = {Preprints},
  author = {Omesh A. Fernando and Joseph Spring and Hannan Xiao},
  title = {CiNeT: A Comparative Study of a CNN-Based Intrusion Detection System with TensorFlow and PyTorch for 5G and Beyond},
  journal = {Preprints}
}
```

---

## 🤝 Contributing
Issues and pull requests are welcome. The codebase is modular and can be extended for custom architectures and monitoring.

## 👥 Authors
Omesh Fernando
---
