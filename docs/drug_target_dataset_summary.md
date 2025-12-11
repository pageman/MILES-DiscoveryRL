# EvE-162 Drug-Target Activity Dataset - Comprehensive Analysis

## Executive Summary

The **EvE-162 Comprehensive Pharmome Atlas** dataset contains **386,969** drug-target activity measurements from the HuggingFace repository. This is a large-scale pharmacological dataset designed for safety, selectivity analysis, and machine learning benchmarking in drug discovery.

---

## Dataset Dimensions

- **Rows**: 386,969 interactions
- **Columns**: 31 features
- **Memory Size**: 479.45 MB
- **Data Completeness**: 84.10%
- **Duplicate Rows**: 0 (100% unique)

---

## Key Statistics

### Compound Coverage
- **Unique Compounds**: 1,397
- **DrugBank Annotated**: 371,180 interactions (95.92%)
- **Progressed Compounds**: 28,161 (7.28%)
- **Most Tested Compound**: Dihydroergotamine Mesylate (277 assays)

### Target Coverage
- **Unique Targets**: 160
- **Unique Genes**: 148
- **Target Classes**: 4 (7TM, NR, Kinase, Viability)
- **Mutant Variants**: 15,367 (3.97%)
- **Most Studied Gene**: BTK (9,779 assays)

### Assay Coverage
- **Unique Assays**: 277
- **Assay Technologies**: 3 (FRET, TR-FRET, Luminescence)
- **Data Releases**: 9 versions (1.2 to 7.0)

---

## Target Class Distribution

| Target Class | Count | Percentage |
|--------------|-------|------------|
| 7TM (GPCRs) | 237,490 | 61.37% |
| NR (Nuclear Receptors) | 89,408 | 23.10% |
| Kinase | 58,674 | 15.16% |
| Viability | 1,397 | 0.36% |

**Insight**: The dataset is heavily weighted toward GPCR targets (7TM), which are the largest class of drug targets, making this dataset particularly valuable for GPCR drug discovery.

---

## Interaction Modes & Mechanisms

### Modes
| Mode | Count | Percentage |
|------|-------|------------|
| Agonist | 163,449 | 42.24% |
| Antagonist | 163,449 | 42.24% |
| Binding | 58,674 | 15.16% |
| Inhibition | 1,397 | 0.36% |

### Mechanisms
| Mechanism | Count | Percentage |
|-----------|-------|------------|
| Barr2 Recruitment | 237,490 | 61.37% |
| Co-factor Recruitment | 89,408 | 23.10% |
| Competition Binding | 58,674 | 15.16% |
| ATP Production | 1,397 | 0.36% |

**Insight**: The dataset provides balanced coverage of agonist vs antagonist modes, enabling comprehensive pharmacological profiling.

---

## Activity & Potency Analysis

### Activity Outcomes
- **Active Compounds**: 31,644 (8.18%)
- **Inactive Compounds**: 355,325 (91.82%)

**Hit Rate**: 8.18% - This is typical for large-scale screening datasets where most compounds don't show activity.

### Potency (pXC50) Statistics
Only 3.03% of measurements (11,730) have quantified potency values.

| Statistic | Value |
|-----------|-------|
| Mean | 6.20 |
| Median | 6.00 |
| Std Dev | 0.96 |
| Min | 5.00 |
| Max | 11.60 |
| Range | 6.60 |
| Skewness | 1.13 |
| Kurtosis | 1.27 |

**Interpretation**:
- pXC50 of 6.0 corresponds to ~1 μM (micromolar) potency
- pXC50 of 11.6 corresponds to ~2.5 pM (picomolar) potency
- Positive skewness indicates more low-potency compounds with a tail of highly potent compounds
- The mean potency is in the micromolar range, typical for screening libraries

### Max Activity Statistics
| Statistic | Value |
|-----------|-------|
| Mean | 6.40% |
| Median | 1.00% |
| Std Dev | 20.11% |
| Min | -176.00% |
| Max | 1,615.60% |

**Insight**: The large range and high standard deviation indicate diverse pharmacological responses, including super-agonists (>100%) and inverse agonists (negative values).

---

## Dose-Response Curve Characteristics

### Slope (Hill Coefficient)
| Statistic | Value |
|-----------|-------|
| Mean | 2.11 |
| Median | 1.20 |
| Range | 0.0 - 51.3 |
| IQR | 0.9 - 1.9 |

**Interpretation**:
- Median slope of 1.2 suggests typical sigmoidal dose-response curves
- High skewness (4.50) indicates presence of super-steep curves
- Values near 1.0 indicate non-cooperative binding
- Higher values suggest positive cooperativity or multi-site interactions

### Asymptotic Parameters

**Asymp Min** (Lower asymptote):
- Mean: -1.25%
- Median: -0.10%
- Range: -100.4% to 49.7%

**Asymp Max** (Upper asymptote):
- Mean: 83.98%
- Median: 93.70%
- Range: 4.5% to 1,615.6%

**Correlation**: asymp_max shows perfect correlation (r=1.0) with outcome_max_activity, as expected.

---

## Assay Technology Distribution

| Technology | Count | Percentage |
|------------|-------|------------|
| FRET | 237,490 | 61.37% |
| TR-FRET | 148,082 | 38.27% |
| Luminescence | 1,397 | 0.36% |

**Insight**: Predominantly uses fluorescence-based technologies, which are standard for high-throughput pharmacological assays.

---

## Data Quality Flags

### Frequency Flag
- **Flagged**: 30,086 (7.77%)
- **Not Flagged**: 356,883 (92.23%)

Likely indicates compounds that appear frequently in the assay results.

### Viability Flag
- **Flagged**: 21,420 (5.54%)
- **Not Flagged**: 365,549 (94.46%)

Indicates potential cytotoxicity or cell viability issues.

### Quantification Status
- **Quantified**: 11,730 (3.03%)
- **Not Quantified**: 375,239 (96.97%)

**Critical Insight**: Only 3% of measurements have full dose-response curves with quantified potency. The remaining 97% are likely single-point or limited dose-response measurements.

---

## Top 10 Most Studied Targets

| Rank | Gene | Assays | Target Name |
|------|------|--------|-------------|
| 1 | BTK | 9,779 | Tyrosine-protein kinase BTK |
| 2 | ALK | 4,191 | ALK tyrosine kinase receptor |
| 3 | EGFR | 4,191 | Epidermal growth factor receptor |
| 4 | DRD2 | 2,794 | D2 dopamine receptor |
| 5 | S1PR1 | 2,794 | Sphingosine 1-phosphate receptor 1 |
| 6 | P2RY14 | 2,794 | P2Y purinoceptor 14 |
| 7 | P2RY1 | 2,794 | P2Y purinoceptor 1 |
| 8 | P2RY12 | 2,794 | P2Y purinoceptor 12 |
| 9 | NPY2R | 2,794 | Neuropeptide Y receptor type 2 |
| 10 | CNR2 | 2,794 | Cannabinoid receptor 2 |

**Insight**: BTK (Bruton's tyrosine kinase) is heavily represented, likely due to its importance in cancer and autoimmune diseases. The dataset includes multiple BTK mutant variants for resistance profiling.

---

## Data Completeness Analysis

### Columns with Missing Data

| Column | Missing Count | Missing % |
|--------|---------------|-----------|
| slope | 375,308 | 96.99% |
| asymp_max | 375,239 | 96.97% |
| outcome_potency_pxc50 | 375,239 | 96.97% |
| asymp_min | 375,239 | 96.97% |
| pxc50_modifier | 375,239 | 96.97% |
| compound__drugbank_id | 15,789 | 4.08% |
| compound__smiles | 4,155 | 1.07% |
| compound__inchikey | 4,155 | 1.07% |
| compound__unii | 3,047 | 0.79% |
| target__gene | 1,397 | 0.36% |

**Key Observation**: The ~97% missing data for dose-response parameters is expected - these fields only apply to the 3% of measurements with full quantified curves.

---

## Data Release History

| Release | Count | Percentage | Cumulative |
|---------|-------|------------|------------|
| 1.2 | 53,086 | 13.72% | 13.72% |
| 2.1 | 58,674 | 15.16% | 28.88% |
| 3.0 | 120 | 0.03% | 28.91% |
| 3.1 | 58,674 | 15.16% | 44.07% |
| 4.0 | 67,216 | 17.37% | 61.44% |
| 5.0 | 218 | 0.06% | 61.50% |
| 5.1 | 67,056 | 17.33% | 78.83% |
| 6.0 | 40,655 | 10.51% | 89.34% |
| 7.0 | 41,270 | 10.66% | 100.00% |

**Insight**: The dataset has grown incrementally, with major releases (4.0, 5.1) contributing the most data.

---

## Statistical Distribution Characteristics

### Numerical Variables Summary

| Variable | Skewness | Kurtosis | CV (%) | Distribution Type |
|----------|----------|----------|--------|-------------------|
| outcome_potency_pxc50 | 1.13 | 1.27 | 15.5% | Right-skewed, leptokurtic |
| outcome_max_activity | 8.48 | 250.49 | 314.4% | Extremely right-skewed |
| observed_max | 6.32 | 74.54 | 302.2% | Extremely right-skewed |
| slope | 4.50 | 33.91 | 128.0% | Extremely right-skewed |
| asymp_min | -2.80 | 20.80 | -636.8% | Left-skewed |
| asymp_max | 9.57 | 275.93 | 49.7% | Extremely right-skewed |
| release | 0.11 | -1.09 | 46.6% | Nearly uniform |

**Interpretation**:
- High kurtosis values indicate heavy-tailed distributions with outliers
- The extreme skewness in activity measurements reflects the nature of drug discovery: most compounds are inactive or weakly active, with rare highly active compounds
- The negative CV for asymp_min indicates the mean is negative (baseline activity)

---

## Key Correlations

| Variable 1 | Variable 2 | Correlation |
|------------|------------|-------------|
| outcome_max_activity | asymp_max | 1.000 |
| outcome_max_activity | observed_max | 0.972 |
| observed_max | asymp_max | 0.785 |

**Insight**: Perfect correlation between max_activity and asymp_max confirms they represent the same measurement (upper asymptote of dose-response curve).

---

## Mutant Target Analysis

- **Total Mutants**: 15,367 (3.97%)
- **Wild-type**: 371,602 (96.03%)

**Applications**:
- Drug resistance profiling
- Understanding mutation effects on drug binding
- Precision medicine research

**Example**: BTK mutants (M437R, C481S, T316A, T474S) are clinically relevant resistance mutations for BTK inhibitors.

---

## Compound Annotation Quality

| Identifier | Coverage | Count |
|------------|----------|-------|
| compound_id | 100% | 1,397 |
| compound__name | 100% | 1,397 |
| compound__cas | 100% | 1,397 |
| compound__drugbank_id | 95.92% | 1,339 unique |
| compound__unii | 99.21% | 1,385 unique |
| compound__smiles | 98.93% | 1,382 unique |
| compound__inchikey | 98.93% | 1,381 unique |

**Insight**: Excellent chemical structure annotation (>98% coverage) enables computational chemistry applications.

---

## Recommended Use Cases

### 1. Machine Learning & AI
- **Target**: 8.18% hit rate provides class-imbalanced classification problem
- **Features**: 31 columns including pharmacological, chemical, and biological descriptors
- **Applications**:
  - Activity prediction models
  - QSAR (Quantitative Structure-Activity Relationship)
  - Multi-task learning across targets
  - Transfer learning for rare targets

### 2. Drug Repurposing
- 96% of compounds have DrugBank IDs (approved/investigational drugs)
- 7.28% compounds progressed in development
- Cross-target activity profiling enables polypharmacology analysis

### 3. Safety & Selectivity Profiling
- Multi-target screening enables selectivity assessment
- Off-target activity detection
- Promiscuity analysis using frequency_flag
- Cytotoxicity assessment using viability_flag

### 4. Target-Specific Analysis
- BTK resistance mutation profiling (5 variants)
- EGFR mutation analysis (clinically relevant)
- ALK inhibitor resistance mechanisms

### 5. Pharmacological Research
- Dose-response curve characterization
- Technology bias assessment (FRET vs TR-FRET)
- Mechanism of action classification
- Cooperativity analysis via Hill slopes

---

## Data Quality Strengths

1. **No Duplicate Rows**: 100% unique measurements
2. **Complete Core Data**: All assay, target, compound IDs present
3. **High Annotation Quality**: >95% DrugBank coverage
4. **Standardized Identifiers**: CAS, UNII, InChIKey provided
5. **Quantified Subset**: 3% with full dose-response curves
6. **Version Control**: Release tracking enables temporal analysis
7. **Mutant Coverage**: 4% mutant targets for resistance studies

---

## Data Quality Considerations

1. **Sparse Potency Data**: Only 3% quantified - most are single-point measurements
2. **Missing Chemical Structures**: 1% lack SMILES (likely mixtures/salts)
3. **Imbalanced Classes**:
   - 92% inactive vs 8% active
   - 61% 7TM vs 15% Kinase
4. **Technology Bias**: Different technologies may have systematic differences
5. **Release Heterogeneity**: Data collected over multiple versions may have batch effects

---

## Statistical Insights

### Potency Distribution
- **pXC50 range**: 5.0 to 11.6
- **Corresponding IC50 range**: 100 μM to 2.5 pM (8 orders of magnitude)
- **Most common potency**: ~1 μM (pXC50 = 6.0)
- **Highly potent compounds** (pXC50 > 9): ~1% of quantified measurements

### Activity Distribution
- **Median activity**: 1% (mostly inactive)
- **Mean activity**: 6.4% (skewed by active compounds)
- **Full agonists** (>80%): Substantial fraction of active compounds
- **Partial agonists** (20-80%): Moderate representation
- **Inverse agonists** (<0%): Present but rare

### Curve Quality
- **Typical Hill slope**: 0.9 - 1.9 (IQR)
- **Cooperative binding**: Present in ~25% (slope > 2)
- **Flat curves** (slope < 0.5): Indicate poor curve quality or unusual pharmacology

---

## Computational Considerations

### Memory Requirements
- **Full dataset**: 479 MB in memory
- **Numerical subset**: ~53 MB
- **Recommended RAM**: 2+ GB for analysis
- **Processing**: Pandas/NumPy handle efficiently

### Performance Metrics
- **Load time**: <10 seconds on standard hardware
- **Row iteration**: 387K rows manageable for most operations
- **Columnar operations**: Optimized for vectorized computation

---

## Comparative Context

### Dataset Scale
- **Large by pharma standards**: 387K measurements is substantial
- **Multi-target**: 160 targets provides good coverage
- **Balanced modes**: Equal agonist/antagonist representation
- **Clinical relevance**: High DrugBank coverage

### Uniqueness
- **Comprehensive**: Covers 4 major target classes
- **Detailed**: Dose-response curves with asymptotic parameters
- **Annotated**: Rich chemical and biological identifiers
- **Versioned**: Release tracking enables reproducibility

---

## Conclusion

The **EvE-162 Drug-Target Activity Dataset** is a high-quality, comprehensive pharmacological resource containing **386,969 drug-target interactions**. With excellent compound annotation (96% DrugBank coverage), diverse target representation (160 targets across 4 classes), and rich dose-response data, this dataset is ideally suited for:

- Machine learning model development
- Drug repurposing studies
- Safety and selectivity profiling
- Target-specific mechanism analysis
- Pharmacological research

The 8.18% hit rate, balanced agonist/antagonist coverage, and inclusion of clinically relevant mutant targets make this a valuable resource for both computational and experimental drug discovery research.

### Key Metrics Summary
- **386,969** total interactions
- **1,397** unique compounds (96% annotated)
- **160** targets across 4 classes
- **8.18%** hit rate
- **84%** data completeness
- **0%** duplicates
- **3%** with quantified dose-response curves

This dataset represents a gold-standard resource for ML benchmarking, safety assessment, and pharmacological discovery in the modern drug development pipeline.
