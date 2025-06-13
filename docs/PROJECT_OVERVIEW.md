# TomatoLeaf-AI: Project Overview

## 🎯 Research Objectives

TomatoLeaf-AI addresses three critical challenges in agricultural AI deployment:

1. **Cross-Domain Generalization**: Bridging the gap between laboratory and field conditions
2. **Severe Class Imbalance**: Handling 75:1 imbalance ratios in real-world datasets
3. **Computational Constraints**: Enabling edge deployment on resource-limited devices

## 🔬 Technical Methodology

### 1. Cross-Domain Benchmark Creation

#### Dataset Unification
- **PlantVillage**: 54,305 images under controlled laboratory conditions
- **TomatoVillage**: 5,000 images from real field environments
- **Combined Dataset**: 15 harmonized disease classes with consistent labeling

#### Class Harmonization Process
```
Original Classes → Unified Classes
├── PlantVillage (10 classes) → 15 unified classes
├── TomatoVillage (8 classes) → 15 unified classes
└── Combined (15 classes) → Standardized taxonomy
```

### 2. Data Preprocessing Pipeline

#### Strategic Data Augmentation
- **Geometric Transformations**: Rotation, flipping, scaling
- **Color Space Augmentation**: Brightness, contrast, saturation
- **Noise Injection**: Gaussian noise for robustness
- **Crop Variations**: Random cropping and resizing

#### ADASYN-Based Balancing
```python
# Class distribution before balancing
Healthy: 18,160 samples
Bacterial_Spot: 2,127 samples  
Early_Blight: 1,000 samples
# ... (75:1 imbalance ratio)

# After ADASYN balancing
All classes: ~15,000 samples each
```

### 3. Ensemble Learning Framework

#### Teacher Ensemble Architecture
- **DenseNet-121**: Dense connectivity for feature reuse
- **ResNet-101**: Deep residual learning for gradient flow
- **DenseNet-201**: Extended dense connections
- **EfficientNet-B4**: Compound scaling for efficiency

#### Soft Voting Strategy
```python
ensemble_prediction = (
    α₁ × P_densenet121 + 
    α₂ × P_resnet101 + 
    α₃ × P_densenet201 + 
    α₄ × P_efficientnet_b4
) / 4
```

### 4. Knowledge Distillation Framework

#### Temperature-Scaled Distillation
```python
# Soft targets from teacher ensemble
soft_targets = softmax(teacher_logits / T)

# Student learning objective
L_KD = KL_divergence(student_logits/T, soft_targets)
L_hard = CrossEntropy(student_logits, true_labels)
L_total = α × L_hard + (1-α) × T² × L_KD
```

#### Feature-Level Distillation
- **Penultimate Layer Matching**: L2 distance between feature representations
- **Attention Transfer**: Spatial attention map alignment
- **Combined Loss**: Logit + Feature + Hard label losses

### 5. Model Quantization Pipeline

#### Post-Training Quantization
- **INT8 Quantization**: 8-bit integer representation
- **Dynamic Quantization**: Runtime weight quantization
- **Static Quantization**: Calibration-based quantization

#### Mobile Optimization
```python
# Quantization workflow
FP32 Model (4.2 MB) → INT8 Model (1.46 MB)
Compression Ratio: 671×
Accuracy Retention: 97.46% (vs 98.53% FP32)
```

## 📊 Experimental Design

### Systematic Evaluation Protocol

#### 1. Baseline Experiments (24 Architectures)
- CNN architectures: ResNet, DenseNet, EfficientNet families
- Vision Transformers: ViT, Swin Transformer variants
- Lightweight models: MobileNet, ShuffleNet, SqueezeNet

#### 2. Hyperparameter Optimization
- **Grid Search**: Learning rate, batch size, optimizer
- **Bayesian Optimization**: Advanced hyperparameter tuning
- **Early Stopping**: Validation-based convergence

#### 3. Knowledge Distillation Experiments (110 Total)
```
Step 1: Basic KD (50 experiments)
├── 10 temperatures × 5 student models

Step 2: Advanced Temperature Scaling (20 experiments)
├── Top 5 teacher temps × 4 student temps

Step 3: Feature-Only Distillation (20 experiments)
├── Same temperature combinations

Step 4: Combined Distillation (20 experiments)
└── Logit + Feature distillation
```

### Cross-Domain Validation

#### Evaluation Datasets
1. **Combined Test Set**: Balanced evaluation across domains
2. **PlantVillage Test**: Laboratory condition validation
3. **TomatoVillage Test**: Field condition validation
4. **PlantDoc**: External validation dataset

#### Performance Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Per-Class Metrics**: Individual disease class performance
- **Confusion Matrix**: Detailed error analysis

## 🧠 Explainable AI Framework

### Attention Visualization
- **Grad-CAM++**: Improved gradient-based attention
- **LIME**: Local interpretable explanations
- **Integrated Gradients**: Attribution-based explanations

### Biological Validation
- **Symptom Localization**: Disease-specific attention patterns
- **Pathology Alignment**: Consistency with plant pathology
- **Expert Validation**: Agricultural expert review

## 📱 Mobile Deployment Architecture

### Flutter Application
- **Cross-Platform**: iOS and Android compatibility
- **Offline Inference**: On-device model execution
- **Multilingual Support**: English, Bengali, Spanish
- **Real-Time Processing**: Camera-based disease detection

### Edge Optimization
```
Model Pipeline:
Raw Image → Preprocessing → Quantized Model → Post-processing → Results
     ↓              ↓              ↓              ↓              ↓
  224×224      Normalization    INT8 Inference   Confidence    Disease Class
```

## 🔄 Reproducibility Framework

### Experiment Tracking
- **Weights & Biases**: Comprehensive experiment logging
- **Configuration Management**: YAML-based parameter tracking
- **Version Control**: Git-based code versioning
- **Environment Management**: Docker containerization

### Standardized Evaluation
- **Fixed Random Seeds**: Reproducible random number generation
- **Consistent Data Splits**: Standardized train/validation/test splits
- **Evaluation Protocols**: Unified metrics and reporting

## 🌍 Real-World Impact

### Agricultural Applications
- **Early Disease Detection**: Preventive crop management
- **Treatment Recommendations**: Targeted intervention strategies
- **Yield Optimization**: Reduced crop losses
- **Farmer Education**: Disease identification training

### Scalability Considerations
- **Multi-Crop Extension**: Framework adaptable to other crops
- **Regional Adaptation**: Local disease variant handling
- **Language Localization**: Multi-language support
- **Hardware Flexibility**: Various mobile device compatibility

## 📈 Performance Benchmarks

### Computational Efficiency
| Model | Parameters | FLOPs | Memory | Inference Time | Accuracy |
|-------|------------|-------|--------|----------------|----------|
| Teacher Ensemble | 163M | 15.2G | 652 MB | 12.6ms | 99.15% |
| ShuffleNetV2 | 1.26M | 146M | 5.1 MB | 0.29ms | 98.53% |
| Quantized INT8 | 1.26M | 146M | 1.46 MB | 0.29ms | 97.46% |

### Cross-Domain Robustness
| Test Dataset | Accuracy | F1-Score | Domain Gap |
|--------------|----------|----------|------------|
| Combined | 99.15% | 97.07% | Baseline |
| PlantVillage | 98.87% | 96.45% | -0.28% |
| TomatoVillage | 95.70% | 93.62% | -3.45% |
| PlantDoc | 92.34% | 89.78% | -6.81% |

## 🔮 Future Directions

### Technical Enhancements
1. **Advanced Distillation**: Progressive knowledge transfer
2. **Neural Architecture Search**: Automated model design
3. **Federated Learning**: Distributed model training
4. **Multi-Modal Fusion**: Image + sensor data integration

### Application Extensions
1. **Multi-Crop Support**: Extend to other agricultural crops
2. **Pest Detection**: Insect and pest identification
3. **Nutritional Analysis**: Nutrient deficiency assessment
4. **Growth Monitoring**: Temporal crop development tracking

---

This project establishes a comprehensive framework for agricultural AI deployment, bridging the gap between research and real-world application while maintaining scientific rigor and practical feasibility. 