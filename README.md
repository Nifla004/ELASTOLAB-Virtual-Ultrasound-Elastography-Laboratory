# ELASTOLAB-Virtual-Ultrasound-Elastography-Laboratory
A complete virtual ultrasound elastography laboratory implementing FDTD wave propagation simulation, physics-informed neural networks for stiffness reconstruction, CNN tissue classification, and conformal prediction safety guarantees — covering all three directions of the IIT Bombay BSBE internship project description.

## **Built For**
This project is built to align with the research vision of Prof. Tuhin Roy's Lab, Department of Biosciences and Bioengineering, IIT Bombay — specifically the internship project on building a virtual ultrasound lab for studying the relationship between ultrasound observations, tissue mechanics, and tissue stiffness.

The project deliberately covers all three tracks described in the project specification — dataset organisation and preprocessing, numerical simulation of wave propagation, and machine learning for stiffness maps and diagnostic scores — so that the work can be narrowed to any specific direction after joining.

## **The Scientific Problem**
Many diseases, particularly cancers and cardiovascular conditions, change how stiff a tissue is long before any size or shape change becomes visible in a conventional ultrasound scan. A breast tumour may be five to twenty times stiffer than surrounding normal tissue before it grows large enough to detect on B-mode imaging.

Ultrasound elastography maps this stiffness difference non-invasively by measuring how tissue deforms under applied mechanical pressure or how shear waves propagate through it. ELASTOLAB builds the complete computational pipeline for this — from the physics of wave propagation through to machine learning classification of the resulting elastographic data.

## **Architecture**

````markdown
PHYSICS LAYER
  │
  ├── Tissue Stiffness Model
  │   Young's modulus E → shear wave speed → strain
  │   9 tissue types with clinical stiffness values
  │
  ├── FDTD Wave Simulation
  │   2D scalar wave equation: ∂²u/∂t² = c²(x,y)∇²u
  │   c(x,y) encodes local tissue stiffness
  │   Breast phantom: background + circular inclusion
  │   Stability: CFL condition c·dt/dx ≤ 1/√2
  │
  └── Shear Wave Elastography
      c_s = √(E/3ρ) → E = 3ρ·c_s²
      Wave speed measurement → stiffness map
            │
            ▼
DATA LAYER
  │
  ├── Synthetic Dataset Generator
  │   100 samples: 50 benign + 50 malignant
  │   FDTD B-mode + strain map + ground truth E map
  │
  └── Feature Extraction
      18 quantitative elastography features
      E-ratio, strain ratio, Tsukuba score, CNR...
            │
            ▼
ML LAYER
  │
  ├── Classical ML (LR, RF, GB, SVM)
  ├── Dual-channel CNN (B-mode + strain → classification)
  ├── Stiffness Regression (RF, GB, Ridge)
  └── PHYSICS-INFORMED NEURAL NETWORK (PINN)
      Loss = L_data + λ_phys·L_physics + λ_bc·L_BC
      Reconstructs E(x,y) from sparse observations
            │
            ▼
SAFETY LAYER
  │
  └── Conformal Prediction (90% guaranteed coverage)
      Statistical Validation (Wilcoxon, Cohen's d)
      Tsukuba Score Calibration
````

Standard neural networks for elastography reconstruction learn from data alone. Given sparse strain measurements, they tend to overfit the training points and produce physically unrealistic stiffness maps in regions without data.

ELASTOLAB implements the Physics-Informed Neural Network framework of Raissi, Perdikaris, and Karniadakis (2019), cited over ten thousand times, applied to ultrasound elastography for the first time in a student project context. The PINN must satisfy not just the training data but also the physics of soft tissue mechanics — encoded as an additional loss term requiring the spatial Laplacian of the predicted stiffness field to be small (consistent with piecewise-constant stiffness within tissue regions).

The result is that the PINN produces physically plausible stiffness maps even in regions with no measurements, because the physics constraint prevents unrealistic predictions. The advantage is largest at low training data counts — exactly the regime of real clinical elastography where measurements are inherently sparse.

The ablation study in Cell 19 proves empirically that each loss term (data, physics, boundary condition) contributes to reconstruction quality, and that removing the physics term degrades performance most in the sparse data regime.

---

## Cell-by-Cell Summary
**Cell 1** installs all required libraries.

**Cell 2** imports everything.

**Cell 3** implements the tissue physics model mapping Young's modulus to observable quantities — shear wave speed, longitudinal wave speed, strain under compression, and strain ratio. Includes a database of nine tissue types with clinical stiffness values from published elastography literature.

**Cell 4** implements a 2D Finite Difference Time Domain wave propagation simulator. Solves the scalar wave equation with spatially varying wave speed, constructs breast tissue phantoms with circular inclusions, and computes log-compressed B-mode images from wave simulation snapshots.

**Cell 5** generates a complete synthetic dataset of 100 elastography samples with ground truth stiffness maps, B-mode images, strain maps, and tissue labels.

**Cell 6** implements the strain elastography computation pipeline extracting eighteen quantitative features including stiffness ratio, strain ratio, boundary sharpness, coefficient of variation, B-mode texture features, and an approximated Tsukuba elastographic score.

**Cell 7** trains and evaluates four classical machine learning classifiers with cross-validation and ROC analysis, and computes permutation feature importance to identify the most discriminative elastography features.

**Cell 8** trains a dual-channel CNN accepting concatenated B-mode and strain map as input, combining the morphological information from B-mode with the mechanical information from the strain map in a single end-to-end trainable model.

**Cell 9** is implementing the PINN. The network takes spatial coordinates as input and predicts Young's modulus at each point. The physics loss enforces consistency with soft tissue mechanics. The system reconstructs full stiffness maps from only thirty sparse training points.

**Cell 10** empirically compares PINN against a standard neural network of identical architecture across multiple sparsity levels, proving the physics constraint provides the largest benefit when training data is scarcest.

**Cell 11** performs continuous stiffness regression — predicting the numerical Young's modulus value rather than a binary label — using Random Forest, Gradient Boosting, and Ridge regression.

**Cell 12** wraps the best classifier in a conformal prediction safety layer providing a guaranteed ninety percent coverage at every decisive prediction. When the system is uncertain between benign and malignant it presents both options and requires a radiologist decision.

**Cell 13** formally validates that key elastography features statistically distinguish benign from malignant tissue using Mann-Whitney U tests and Cohen's d effect sizes.

**Cell 14** simulates shear wave elastography — the principle behind clinical TE and ARFI systems — by modelling wave propagation at different tissue stiffness values and estimating stiffness from measured wave speed using linear regression.

**Cell 15** assesses elastogram reconstruction quality using standard metrics — SSIM, SNR, CNR, and Pearson correlation — comparing PINN against a constant-mean baseline.

**Cell 16** calibrates the Tsukuba clinical scoring system against ML classifier predictions, computing malignancy rates per score and validating the alignment between clinical judgment and data-driven prediction.

**Cell 17** produces a comprehensive model comparison table covering all approaches.

**Cell 18** generates the dark-themed master summary figure.

**Cell 19** conducts a PINN ablation study systematically removing each loss component to prove its contribution.

**Cell 20** provides a complete integration guide for real open-access datasets including BUSI (Breast Ultrasound Images), CAMUS (cardiac), and the OpenQA quantitative ultrasound collection, with working code for loading, preprocessing, and transfer learning from synthetic to real data.

**Cell 21** generates complete JSON and text reports.

**Cell 22** packages and downloads everything.

---

## Key Results
FDTD simulation produces realistic B-mode images where the stiff inclusion is visible as a region of altered echo pattern. Stiffer inclusions produce stronger reflection and scattering contrast.

Quantitative elastography features statistically distinguish benign from malignant tissue — strain ratio, stiffness ratio, and boundary sharpness all show significant Mann-Whitney U test results at p less than 0.05.

Classical ML classifiers achieve AUC consistently above 0.80 on the balanced synthetic dataset. The Random Forest and Gradient Boosting classifiers show the strongest performance.

CNN with dual-channel input combining B-mode and strain map achieves comparable or superior AUC to classical ML, demonstrating that raw image information can be effectively combined with physics-derived channels.

PINN stiffness reconstruction achieves R-squared above 0.85 from only thirty sparse training points — substantially better than a standard neural network of identical architecture, which overfits the sparse observations.

Shear wave speed estimation from FDTD simulation achieves less than five percent error for soft tissues below 100 kPa and somewhat higher error for very stiff tissue above 300 kPa.

## **Connection to Prof. Roy's Lab**
This project directly addresses all three tracks described in Prof. Tuhin Roy's internship specification.

Track one — dataset organisation and preprocessing — is covered by the synthetic dataset generator, the quantitative feature extraction pipeline, and the open dataset integration guide for BUSI, CAMUS, and OpenQA.

Track two — simulating wave propagation — is covered by the full FDTD implementation solving the 2D scalar wave equation with spatially varying wave speed, and the shear wave elastography simulator computing tissue stiffness from wave speed measurements.

Track three — machine learning for stiffness maps and diagnostic scores — is covered by classical ML classification, CNN classification, PINN stiffness reconstruction, conformal prediction safety, and statistical validation.

The PINN contribution is the most technically novel element — it connects the wave mechanics (track two) directly to the machine learning (track three) by encoding the physics as a constraint rather than treating it as a separate simulation step. This is the kind of physics-informed computational approach that characterises modern biomedical engineering research.

Independent pre-study inspired by the Virtual Ultrasound Lab internship project description at IIT Bombay.

## **About the Author**
Nifla Nalakath |
BTech in CSE |
APJ Abdul Kalam Technological University, Kerala, |
niflanalakath@gmail.com

