# LiteMatting Training Project

A comprehensive implementation of the LiteMatting model for high-quality image matting (background removal) optimized for mobile deployment.

## 🚀 Features

- **Mobile-Optimized Architecture**: Efficient MobileNet-based backbone
- **State-of-the-Art Components**: MSLPPM, GFNB modules for superior performance
- **Comprehensive Training**: Multi-loss training with Adobe Composition-1K dataset
- **Easy Deployment**: Multiple export formats (PyTorch, TorchScript, ONNX)
- **Google Colab Ready**: Optimized for GPU training in Colab

## 📁 Project Structure

```
LiteMatting/
├── LiteMatting_Training.ipynb  # Main training notebook for Google Colab
├── network.py                  # Model architecture
├── dataset.py                  # Dataset loading and preprocessing
├── losses.py                   # Loss functions and metrics
├── train.py                    # Training script (alternative to notebook)
├── inference.py                # Standalone inference script
├── requirements.txt            # Python dependencies
├── test.py                     # Original test script
└── README.md                   # This file
```

## 🛠 Setup Instructions

### For Google Colab (Recommended)

1. **Upload Project**: Upload this entire folder to Google Colab or your Google Drive

2. **Open Notebook**: Open `LiteMatting_Training.ipynb` in Google Colab

3. **Enable GPU**: Go to Runtime → Change runtime type → Hardware accelerator → GPU

4. **Run All Cells**: Execute the notebook cells in order

### For Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**: Get Adobe Composition-1K dataset (requires academic access)

3. **Train Model**:
   ```bash
   python train.py
   ```

4. **Run Inference**:
   ```bash
   python inference.py --input image.jpg --output alpha.png
   ```

## 📊 Dataset

This project uses the **Adobe Composition-1K** dataset, the standard benchmark for image matting.

### Getting the Dataset

1. Visit: https://sites.google.com/view/deepimagematting
2. Request access with your academic email (student account)
3. Download and extract the dataset
4. Update the `DATASET_PATH` in the notebook

### Expected Directory Structure

```
Composition-1K/
├── Training_set/
│   ├── fg/           # Foreground images
│   ├── alpha/        # Alpha mattes (ground truth)
│   ├── bg/           # Background images
│   ├── merged/       # Composite images
│   └── trimaps/      # Trimap annotations
└── Test_set/
    ├── fg/
    ├── alpha/
    ├── bg/
    ├── merged/
    └── trimaps/
```

## 🏗 Model Architecture

The LiteMatting model consists of:

- **Backbone**: MobileNet-based encoder for efficiency
- **MSLPPM**: Multi-Scale Local Pyramid Pooling Module for multi-scale features
- **GFNB**: Global Feature Network Block for long-range context
- **Decoder**: Progressive upsampling with skip connections

### Key Features:
- **Parameters**: ~2.3M (mobile-friendly)
- **Input**: Image + Trimap + Click guidance
- **Output**: High-quality alpha matte
- **Resolution**: Supports up to 1024x1024 images

## 📈 Training Configuration

### Default Hyperparameters:
- **Input Size**: 512x512
- **Batch Size**: 4 (Colab) / 8 (Local with sufficient GPU memory)
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW with cosine annealing
- **Epochs**: 50-100
- **Loss Weights**: Alpha=1.0, Composition=1.0, Gradient=1.0, Laplacian=1.0

### Loss Functions:
1. **Alpha Loss**: L1 loss in unknown regions
2. **Composition Loss**: L1 loss on composited images
3. **Gradient Loss**: Preserves fine details
4. **Laplacian Loss**: Ensures smoothness

## 📊 Evaluation Metrics

- **SAD (Sum of Absolute Differences)**: Lower is better
- **MSE (Mean Squared Error)**: Lower is better
- **Gradient Error**: Measures detail preservation

## 🚀 Inference

### Using the Jupyter Notebook:
Run the inference cells in `LiteMatting_Training.ipynb`

### Using the Standalone Script:
```bash
# Basic inference
python inference.py --input image.jpg --output alpha.png

# With custom trimap
python inference.py --input image.jpg --trimap trimap.png --output alpha.png

# Create composite with new background
python inference.py --input image.jpg --output alpha.png --composite result.jpg --background 255 0 0
```

### Trimap Options:
- `auto`: Automatic trimap generation
- `center`: Center-based trimap
- `path/to/trimap.png`: Custom trimap file

## 📦 Model Export

The training notebook exports models in multiple formats:

1. **PyTorch Weights** (`.pth`): For PyTorch applications
2. **TorchScript** (`.pt`): For production deployment
3. **ONNX** (`.onnx`): For cross-platform deployment
4. **Full Model** (`.pth`): Complete model with architecture

## 🎯 Performance Tips

### Memory Optimization:
- Reduce batch size if encountering CUDA OOM errors
- Use gradient checkpointing for larger models
- Enable mixed precision training with `torch.cuda.amp`

### Training Tips:
- Start with smaller input size (320x320) then increase
- Adjust loss weights based on your specific use case
- Use data augmentation to improve generalization
- Monitor validation metrics to prevent overfitting

### Inference Optimization:
- Use TorchScript for faster inference
- Consider model quantization for mobile deployment
- Batch multiple images for better GPU utilization

## 🔍 Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use smaller input resolution
   - Enable gradient checkpointing

2. **Dataset Not Found**:
   - Check `DATASET_PATH` variable
   - Ensure dataset structure matches expected format
   - Verify file permissions

3. **Poor Results**:
   - Check if model weights loaded correctly
   - Verify trimap quality
   - Ensure proper data preprocessing

4. **Slow Training**:
   - Enable GPU in Colab
   - Use larger batch size if memory allows
   - Consider reducing input resolution temporarily

## 📚 References

- [LFPNet Paper](https://arxiv.org/abs/2109.12252) - Inspiration for this work
- [Adobe Image Matting Dataset](https://sites.google.com/view/deepimagematting)
- [Papers With Code - Image Matting](https://paperswithcode.com/task/image-matting)

## 📄 License

This project is for academic and research purposes. Please respect the Adobe Composition-1K dataset license terms.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this implementation.

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Review the notebook outputs for error messages
3. Ensure all dependencies are properly installed
4. Verify dataset path and structure

---

Happy matting! 🎨✨
