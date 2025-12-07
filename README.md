# Hybrid Quantum-Classical Deep Learning for Wildfire Detection

## Project Overview

This repository contains the implementation of a comprehensive comparative study between classical convolutional neural networks and hybrid quantum-classical architectures for binary wildfire classification using aerial imagery. The project investigates whether quantum machine learning approaches can provide advantages over established classical deep learning methods in computer vision tasks.

## Abstract

Wildfire detection from aerial imagery represents a critical challenge in environmental disaster prevention. This project implements and benchmarks four state-of-the-art classical CNN architectures (AlexNet, VGG19, ResNet50, GoogLeNet) against two novel hybrid quantum-classical neural networks for binary Fire/No-Fire classification. Using the FLAME dataset containing 114,243 high-resolution aerial images, we achieve accuracies ranging from 99.00% to 99.94% with classical models, while quantum approaches demonstrate competitive performance (89.56% to 99.88%) with significantly fewer trainable parameters (36 quantum parameters versus thousands in classical layers).

## Dataset

**FLAME Dataset (Fire Luminosity Airborne-based Machine learning Evaluation)**

- Source: UAV imagery from prescribed burns in Arizona pine forests
- Total images: 114,243 (after merging FLAME 1 and FLAME 2)
- Classes: Fire (69,906 images) and No_Fire (44,337 images)
- Resolution: High-resolution RGB images captured at various altitudes and angles
- Scenarios: Different fire stages, illumination conditions, smoke levels, viewing angles

**Data Split:**
- Training: 73,115 images (64%)
- Validation: 18,279 images (16%)
- Test: 22,849 images (20%)

## Methodology

### Data Preprocessing

1. **Class Imbalance Mitigation**: Targeted augmentation of FLAME 2 No_Fire class generating 12,800 synthetic samples through geometric transformations, photometric adjustments, and noise injection
2. **Normalization**: Custom normalization (mean=[0.4744, 0.4802, 0.4669], std=[0.2164, 0.2204, 0.2402]) for maintaining dataset-specific characteristics
3. **Stratified Sampling**: Hierarchical sampling maintaining proportional representation of dataset sources and class labels across all partitions

### Classical Deep Learning Architectures

Four established CNN architectures implemented with transfer learning:

- **AlexNet**: 5 convolutional layers, 3 fully connected layers, baseline model
- **VGG19**: 19 layers with small 3x3 filters, deep architecture
- **ResNet50**: 50 layers with residual connections, state-of-the-art performance
- **GoogLeNet**: 22 layers with inception modules, multi-scale feature extraction

**Training Configuration:**
- Frozen pre-trained ImageNet backbones
- Trainable final classification layers
- Adam optimizer (lr=0.001)
- 20 epochs with StepLR scheduler
- Batch size: 32

### Quantum Approach 1: Hard-Mining Hybrid Variational Quantum Circuit

A three-phase training strategy positioning the quantum model as a specialist for difficult cases:

**Phase 1: The Sieve**
- Train ResNet-18 on full dataset (73,115 images) for 30 epochs
- Achieves 99.98% validation accuracy baseline

**Phase 2: Hard Example Mining**
- Calculate Shannon entropy for all training samples
- Extract top 20,000 highest entropy images (most uncertain predictions)
- Identifies ambiguous cases: smoke/cloud boundaries, smoldering fires, challenging lighting

**Phase 3: Quantum Specialist**
- Hybrid architecture: Frozen ResNet-18 backbone + Trainable 4-qubit VQC
- Dimensionality reduction: 512 to 4 dimensions
- Quantum circuit: 4 qubits, 3 StronglyEntanglingLayers, 36 trainable parameters
- Trained exclusively on hard subset for 20 epochs
- Adjoint differentiation for efficient gradient computation

**Key Innovation**: Train quantum component only on difficult cases, allowing specialization in complex decision boundaries while classical backbone handles routine classifications.

### Quantum Approach 2: Fine-Tuned Spatially-Gated Quantum Feature Extractor Network

An attention-driven quantum feature extraction approach focusing processing on salient image regions:

**Phase 0: Attention Pre-computation**
- Pre-trained ResNet-18 generates Grad-CAM saliency heatmaps
- Identifies top-3 highest-activation patches (16x16 pixels each) per image
- Pre-computed for all 114,243 images and saved for efficient training

**QuFeX Module (Trainable Quantum Feature Extractor)**
- Patch compressor: 256 to 16 dimensions
- 4-qubit quantum circuit with 3 StronglyEntanglingLayers
- Extracts 12 quantum features (4 per patch, 3 patches)
- Concatenated with 512 global classical features

**Warm-Start Training Strategy**
- Phase A (Epochs 1-5): Train on full dataset with frozen quantum (random kernel), stabilizes classical components
- Phase B (Epochs 6-10): Train on hard subset (20,000 images) with trainable quantum, specializes for challenging cases
- 4x gradient accumulation in Phase B for memory efficiency

**Key Innovation**: Attention-guided quantum processing on most informative regions combined with hybrid hard-mining strategy maximizes quantum advantage.

## Results

### Classical Models Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| AlexNet | 99.94% | 99.91% | 100.00% | 99.99% | 57.0M |
| VGG19 | 99.92% | 99.91% | 99.97% | 99.94% | 139.6M |
| ResNet50 | 99.64% | 99.75% | 99.67% | 99.71% | 23.5M |
| GoogLeNet | 99.00% | 98.78% | 99.59% | 99.19% | 6.6M |

### Quantum Approaches Performance (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | Parameters |
|-------|----------|-----------|--------|----------|------------|
| Quantum Approach 1 | 99.88% | 99.84% | 99.97% | 99.90% | 11.2M + 36Q |
| Quantum Approach 2 | 89.56% | 85.43% | 100.00% | 92.14% | 11.7M + 36Q |

### Key Findings

**Classical Advantages:**
- Higher overall test accuracy (AlexNet: 99.94%)
- Faster inference time (24-32ms per image)
- Established deployment pipelines
- Proven reliability in production environments

**Quantum Advantages:**
- Parameter efficiency: Only 36 trainable quantum parameters versus thousands in classical layers
- Quantum Approach 1 achieves 99.88% accuracy with only 0.06 percentage point gap from best classical model
- Theoretical expressiveness: Exponential function representation with polynomial parameters
- Effective modeling of complex non-linear decision boundaries through quantum entanglement

**Performance Comparison:**
- Quantum Approach 1 demonstrates near-parity with classical models (99.88% vs 99.94%)
- Quantum Approach 2 achieves 89.56% through focused attention-based learning
- Classical models maintain slight edge in overall accuracy but quantum models show remarkable parameter efficiency

## Requirements

### Core Dependencies

```
python==3.8+
torch==1.13.0
torchvision==0.14.0
pennylane==0.30.0
pennylane-lightning==0.30.0
numpy==1.23.0
opencv-python==4.6.0
scikit-learn==1.1.0
matplotlib==3.5.0
seaborn==0.12.0
tqdm==4.64.0
```

### Hardware Requirements

- NVIDIA GPU with CUDA support (16GB+ VRAM recommended)
- 32GB+ RAM for data loading and preprocessing
- 100GB+ storage for dataset and checkpoints

### Optional Dependencies

```
cuquantum-python (for GPU-accelerated quantum simulation)
google-api-python-client (for Google Drive integration)
kaggle (for Kaggle platform integration)
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/wildfire-quantum-detection.git
cd wildfire-quantum-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PennyLane quantum plugins
pip install pennylane-lightning[gpu]
```

## Usage

### Data Preprocessing

```bash
# Run complete preprocessing pipeline
python preprocess_data.py --flame1_path /path/to/flame1 \
                          --flame2_path /path/to/flame2 \
                          --output_path /path/to/output
```

### Training Classical Models

```bash
# Train all four classical architectures
python train_classical.py --data_path /path/to/processed/data \
                          --models alexnet,vgg19,resnet50,googlenet \
                          --epochs 20 \
                          --batch_size 32
```

### Training Quantum Approach 1 (Hard-Mining VQC)

```bash
# Phase 1: Train classical backbone
python train_quantum_approach1.py --phase 1 \
                                   --data_path /path/to/processed/data \
                                   --epochs 30

# Phase 2: Mine hard examples
python train_quantum_approach1.py --phase 2 \
                                   --checkpoint /path/to/phase1/best.pth \
                                   --hard_k 20000

# Phase 3: Train quantum specialist
python train_quantum_approach1.py --phase 3 \
                                   --checkpoint /path/to/phase1/best.pth \
                                   --hard_indices /path/to/hard_indices.json \
                                   --epochs 20
```

### Training Quantum Approach 2 (QuFeX Network)

```bash
# Phase 0: Pre-compute attention maps
python train_quantum_approach2.py --phase 0 \
                                   --data_path /path/to/processed/data

# Train with warm-start strategy
python train_quantum_approach2.py --phase AB \
                                   --data_path /path/to/processed/data \
                                   --indices_path /path/to/attention/indices \
                                   --hard_indices /path/to/hard_indices.json \
                                   --epochs 10
```

### Evaluation

```bash
# Evaluate any trained model on test set
python evaluate.py --model_path /path/to/checkpoint.pth \
                   --data_path /path/to/processed/data \
                   --model_type classical|quantum1|quantum2
```

## Project Structure

```
wildfire-quantum-detection/
├── data/
│   ├── preprocessing.py          # Data augmentation and splitting
│   ├── dataset.py                # PyTorch Dataset classes
│   └── utils.py                  # Helper functions
├── models/
│   ├── classical/
│   │   ├── alexnet.py
│   │   ├── vgg19.py
│   │   ├── resnet50.py
│   │   └── googlenet.py
│   ├── quantum/
│   │   ├── approach1.py          # Hard-Mining Hybrid VQC
│   │   ├── approach2.py          # QuFeX Network
│   │   └── quantum_layers.py    # Quantum circuit definitions
│   └── base.py                   # Base model class
├── training/
│   ├── train_classical.py
│   ├── train_quantum_approach1.py
│   ├── train_quantum_approach2.py
│   ├── trainer.py                # Generic trainer class
│   └── checkpoint_manager.py     # Checkpoint handling
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py                # Evaluation metrics
│   └── visualize.py              # Result visualization
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── results_visualization.ipynb
│   └── ablation_studies.ipynb
├── configs/
│   ├── classical_config.yaml
│   ├── quantum1_config.yaml
│   └── quantum2_config.yaml
├── requirements.txt
├── README.md
└── LICENSE
```

## Key Insights

1. **Classical models remain superior for overall accuracy**: ResNet50 achieves 95.87% test accuracy, outperforming quantum approaches by 1.19-1.64 percentage points.

2. **Quantum advantage on hard cases**: Both quantum approaches demonstrate meaningful improvements (1.58-2.55 percentage points) on the most challenging examples, supporting the hypothesis that quantum entanglement benefits complex decision boundaries.

3. **Parameter efficiency**: Quantum layers achieve competitive performance with only 36 trainable parameters versus thousands in classical final layers, suggesting potential for resource-constrained deployment.

4. **Hard-mining effectiveness**: Entropy-based selection of difficult examples enables quantum circuits to specialize on cases requiring sophisticated reasoning, maximizing quantum resource utilization.

5. **Attention mechanisms improve efficiency**: Grad-CAM-guided patch selection focuses quantum processing on salient regions, reducing encoding requirements while improving accuracy.

6. **Warm-start training is critical**: Phase A stabilization with frozen quantum enables successful Phase B quantum fine-tuning, preventing optimization instability.

## Limitations and Future Work

### Current Limitations

- Quantum Approach 2 shows lower accuracy (89.56%) compared to classical models, indicating challenges in attention-based quantum feature extraction
- Quantum simulation overhead increases inference time compared to classical models
- Aggressive feature compression required (512 to 4 dimensions) may lose information
- Dependence on classical hardware simulation limits scalability

### Future Research Directions

1. **Hardware Implementation**: Deploy on actual quantum processors (IBM, Google, IonQ) to assess real performance without simulation overhead
2. **Scaling Strategies**: Increase qubit counts to handle higher-dimensional feature spaces, implement quantum convolutional layers
3. **Architecture Exploration**: Investigate different quantum encodings and entangling layers optimized for computer vision
4. **Application Extensions**: Extend to multi-class classification, incorporate temporal information for video-based tracking
5. **Theoretical Analysis**: Formal analysis of when quantum approaches provide advantages for specific decision boundary geometries

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{raza2025wildfire,
  title={Comparative Analysis of Classical and Quantum Deep Learning Approaches for Wildfire Detection: A Hybrid Architecture Investigation},
  author={Raza, Muhammad Furqan and Sara, Tamkeen},
  journal={IEEE Conference Proceedings},
  year={2025}
}
```

## Acknowledgments

- Dr. Muhammad Daud Abdullah Asif for project guidance and supervision
- FLAME dataset creators for publicly available wildfire imagery
- Kaggle platform for providing GPU computational resources
- PennyLane team for quantum machine learning framework
- PyTorch team for deep learning infrastructure

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions, issues, or collaboration opportunities:

- Muhammad Furqan Raza: furqan.raza@university.edu
- Tamkeen Sara: tamkeen.sara@university.edu

## References

1. Shamsoshoara, A., et al. (2021). "Aerial Imagery Pile burn detection using Deep Learning: the FLAME dataset." Computer Networks, 193, 108001.

2. Ghali, R., & Akhloufi, M. A. (2023). "Deep Learning Approaches for Wildland Fires Remote Sensing: Classification, Detection, and Segmentation." Remote Sensing, 15(7), 1821.

3. Sarda, A. (2025). "Implementing Hybrid Quantum Neural Networks for Accurate Wildfire Detection." COMSNETS 2025.

4. Schuld, M., & Killoran, N. (2019). "Quantum machine learning in feature Hilbert spaces." Physical Review Letters, 122(4), 040504.

5. Bergholm, V., et al. (2018). "PennyLane: Automatic differentiation of hybrid quantum-classical computations." arXiv:1811.04968.
