# Systematic Data Augmentation Experiments

## Overview
This framework implements a comprehensive ablation study of data augmentation techniques for tomato leaf disease classification, designed specifically for scientific research and paper publication.

## Research Methodology

### Experimental Design
The experiments follow a systematic approach to evaluate the impact of different data augmentation strategies:

1. **Baseline Comparison**: No augmentation (fair baseline)
2. **Individual Analysis**: Each augmentation technique tested separately
3. **Group Analysis**: Augmentations grouped by type (geometric, photometric, noise/blur)
4. **Combination Analysis**: Testing combinations of groups
5. **Parameter Analysis**: Testing different parameter ranges for key techniques

### Augmentation Categories

#### 1. Geometric Transformations
- **Random Horizontal Flip**: p=0.5
- **Random Vertical Flip**: p=0.5  
- **Random Rotation**: ±10°, ±20°, ±30°
- **Random Scale/Crop**: (0.9,1.1), (0.8,1.2), (0.7,1.3)

#### 2. Photometric Transformations
- **Brightness**: ±10%, ±20%, ±30%
- **Contrast**: ±10%, ±20%, ±30%
- **Saturation**: ±10%, ±20%, ±30%
- **Hue**: ±5%, ±10%, ±15%

#### 3. Noise & Blur Transformations
- **Gaussian Blur**: σ=(0.1,1.0), (0.1,2.0), (0.1,3.0)
- **Gaussian Noise**: std=0.02, 0.05, 0.1

## Experiment Structure

### Total Experiments
The framework generates **~50+ individual experiments** organized as follows:

1. **Baseline**: 1 experiment (no augmentation)
2. **Geometric Individual**: 8 experiments (4 techniques × multiple parameters)
3. **Photometric Individual**: 12 experiments (4 techniques × 3 parameter values each)
4. **Noise/Blur Individual**: 6 experiments (2 techniques × 3 parameter values each)
5. **Group Combinations**: 3 experiments (best parameters from individual analysis)
6. **Two-Group Combinations**: 3 experiments
7. **All Combined**: 1 experiment

### Experiment Groups (in execution order)
1. `baseline` - Control group with no augmentation
2. `geometric_individual` - Individual geometric transformations
3. `photometric_individual` - Individual photometric transformations
4. `noise_individual` - Individual noise/blur transformations
5. `geometric_combined` - Best geometric transformations combined
6. `photometric_combined` - Best photometric transformations combined
7. `noise_combined` - Best noise/blur transformations combined
8. `two_group_combination` - Combinations of two groups
9. `all_combined` - All optimal transformations combined

## Usage

### Run All Experiments (Recommended for Research)
```bash
python run_augmentation_experiments.py --model shufflenet_v2 --epochs 50 --dataset combined
```

### Run Specific Experiment Group
```bash
python run_augmentation_experiments.py --group baseline --model shufflenet_v2
python run_augmentation_experiments.py --group geometric_individual --model shufflenet_v2
python run_augmentation_experiments.py --group photometric_individual --model shufflenet_v2
```

### Run Individual Experiment
```bash
python train_augmentation.py \
    --model shufflenet_v2 \
    --dataset combined \
    --epochs 50 \
    --augmentation_config path/to/config.json \
    --experiment_name "rotation_20deg"
```

## Scientific Reporting Structure

### For Research Papers
The experimental design provides clear structure for academic reporting:

#### 1. Methodology Section
- **Baseline**: No augmentation for fair comparison
- **Ablation Study**: Individual techniques tested separately
- **Parameter Optimization**: Multiple parameter values tested
- **Combination Analysis**: Systematic combination of effective techniques

#### 2. Results Section Organization
```
Table 1: Baseline Performance (No Augmentation)
Table 2: Individual Geometric Transformations
Table 3: Individual Photometric Transformations  
Table 4: Individual Noise/Blur Transformations
Table 5: Group-wise Combinations
Table 6: Best Combined Results
```

#### 3. Statistical Analysis
- Each experiment includes confidence intervals
- Statistical significance testing between groups
- Performance improvement quantification
- Parameter sensitivity analysis

### Key Metrics Tracked
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Per-class and macro-averaged F1-scores
- **Precision/Recall**: Per-class metrics
- **Training Stability**: Loss convergence analysis
- **Computational Cost**: Training time per epoch

## Expected Outcomes

### Research Questions Answered
1. **Which augmentation techniques are most effective for tomato leaf disease classification?**
2. **What are the optimal parameter ranges for each technique?**
3. **How do different augmentation groups interact when combined?**
4. **What is the trade-off between augmentation complexity and performance gain?**

### Performance Expectations
- **Baseline**: ~92.12% (ShuffleNetV2 without augmentation)
- **Individual Techniques**: 1-3% improvement for best techniques
- **Combined Techniques**: 3-5% improvement over baseline
- **Optimal Configuration**: Target 95-97% accuracy

## Timeline & Resources

### Estimated Runtime
- **Individual Experiments**: ~2-3 hours each
- **Total Runtime**: ~100-150 hours for all experiments
- **Recommended**: Run overnight or in batches

### Resource Requirements
- **GPU Memory**: 8GB+ recommended
- **Storage**: ~50GB for all experiment outputs
- **CPU**: Multi-core for data loading efficiency

## Results Analysis

### Automated Analysis Tools
The framework provides:
- **Results Summary**: JSON file with all experiment results
- **Performance Comparison**: Automatic ranking of techniques
- **Statistical Analysis**: Significance testing between methods
- **Visualization**: Plots for paper figures

### Output Structure
```
outputs/augmentation_experiments/
├── augmentation_experiments_summary.json
├── baseline/
│   └── shufflenet_v2_baseline_*/
├── geometric_individual/
│   ├── shufflenet_v2_rotation_10deg_*/
│   ├── shufflenet_v2_rotation_20deg_*/
│   └── ...
├── photometric_individual/
│   ├── shufflenet_v2_brightness_0.1_*/
│   ├── shufflenet_v2_brightness_0.2_*/
│   └── ...
└── [other groups]/
```

## Best Practices for Research

### 1. Reproducibility
- Fixed random seeds across all experiments
- Identical training configurations
- Saved augmentation configurations for each experiment

### 2. Fair Comparison
- Same baseline model architecture
- Same training hyperparameters
- Same evaluation protocol

### 3. Statistical Rigor
- Multiple runs for statistical significance (optional)
- Confidence interval reporting
- Appropriate statistical tests

### 4. Documentation
- Complete experiment logs
- Parameter configurations saved
- Code snapshots for reproducibility

## Integration with Existing Pipeline

This augmentation study can be combined with:
- **Knowledge Distillation**: Apply best augmentations to KD experiments
- **Model Comparison**: Test augmentations across different architectures
- **Ensemble Methods**: Use augmented models in ensemble combinations

## Citation and Reporting

For academic papers, this methodology supports:
- Comprehensive ablation studies
- Statistical significance reporting
- Parameter optimization documentation
- Reproducible research standards

The systematic approach ensures all experiments are methodologically sound and suitable for peer-reviewed publication. 