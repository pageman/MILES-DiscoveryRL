#!/usr/bin/env python3
"""
Comprehensive Descriptive Statistics Analysis for Drug-Target Activity Dataset
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("="*80)
print("DRUG-TARGET ACTIVITY DATASET - COMPREHENSIVE DESCRIPTIVE STATISTICS")
print("="*80)
print("\nLoading dataset...")

df = pd.read_csv("/Users/paulamerigojr.iipajo/Downloads/Drug Target Activity.csv")

print(f"✓ Dataset loaded successfully!")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 1. BASIC DATASET OVERVIEW
# ============================================================================
print("\n" + "="*80)
print("1. DATASET OVERVIEW")
print("="*80)

print("\n1.1 Column Information:")
print("-" * 80)
info_dict = {
    'Column': df.columns,
    'Non-Null Count': [df[col].count() for col in df.columns],
    'Null Count': [df[col].isnull().sum() for col in df.columns],
    'Null %': [f"{df[col].isnull().sum() / len(df) * 100:.2f}%" for col in df.columns],
    'Dtype': [df[col].dtype for col in df.columns],
    'Unique Values': [df[col].nunique() for col in df.columns]
}
info_df = pd.DataFrame(info_dict)
print(info_df.to_string(index=False))

print("\n1.2 Data Types Summary:")
print("-" * 80)
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

# ============================================================================
# 2. NUMERICAL COLUMNS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. NUMERICAL COLUMNS - DESCRIPTIVE STATISTICS")
print("="*80)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nFound {len(numerical_cols)} numerical columns: {', '.join(numerical_cols)}")

if numerical_cols:
    print("\n2.1 Comprehensive Statistics:")
    print("-" * 80)

    stats_df = df[numerical_cols].describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99])
    print(stats_df.to_string())

    print("\n2.2 Additional Statistical Measures:")
    print("-" * 80)

    additional_stats = pd.DataFrame({
        'Column': numerical_cols,
        'Skewness': [df[col].skew() for col in numerical_cols],
        'Kurtosis': [df[col].kurtosis() for col in numerical_cols],
        'Variance': [df[col].var() for col in numerical_cols],
        'Range': [df[col].max() - df[col].min() for col in numerical_cols],
        'IQR': [df[col].quantile(0.75) - df[col].quantile(0.25) for col in numerical_cols],
        'CV (%)': [df[col].std() / df[col].mean() * 100 if df[col].mean() != 0 else np.nan for col in numerical_cols]
    })
    print(additional_stats.to_string(index=False))

# ============================================================================
# 3. CATEGORICAL COLUMNS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. CATEGORICAL COLUMNS - DISTRIBUTION ANALYSIS")
print("="*80)

categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
print(f"\nFound {len(categorical_cols)} categorical columns")

for col in categorical_cols:
    unique_count = df[col].nunique()
    null_count = df[col].isnull().sum()

    print(f"\n3.{categorical_cols.index(col)+1} {col}")
    print("-" * 80)
    print(f"  Unique values: {unique_count:,}")
    print(f"  Null values: {null_count:,} ({null_count/len(df)*100:.2f}%)")

    if unique_count <= 20:
        print(f"  Value counts:")
        value_counts = df[col].value_counts(dropna=False)
        for val, count in value_counts.items():
            pct = count / len(df) * 100
            print(f"    {val}: {count:,} ({pct:.2f}%)")
    else:
        print(f"  Top 15 most frequent values:")
        top_values = df[col].value_counts(dropna=False).head(15)
        for val, count in top_values.items():
            pct = count / len(df) * 100
            val_str = str(val)[:50] + "..." if len(str(val)) > 50 else str(val)
            print(f"    {val_str}: {count:,} ({pct:.2f}%)")

# ============================================================================
# 4. BOOLEAN COLUMNS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. BOOLEAN COLUMNS - DISTRIBUTION")
print("="*80)

bool_cols = [col for col in df.columns if df[col].dtype == 'bool' or set(df[col].dropna().unique()).issubset({True, False})]
print(f"\nFound {len(bool_cols)} boolean columns")

for col in bool_cols:
    true_count = df[col].sum()
    false_count = len(df) - true_count - df[col].isnull().sum()
    null_count = df[col].isnull().sum()

    print(f"\n  {col}:")
    print(f"    True:  {true_count:,} ({true_count/len(df)*100:.2f}%)")
    print(f"    False: {false_count:,} ({false_count/len(df)*100:.2f}%)")
    if null_count > 0:
        print(f"    Null:  {null_count:,} ({null_count/len(df)*100:.2f}%)")

# ============================================================================
# 5. KEY DOMAIN-SPECIFIC ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. DOMAIN-SPECIFIC ANALYSIS")
print("="*80)

# 5.1 Target Analysis
print("\n5.1 Target Analysis:")
print("-" * 80)
if 'target__class' in df.columns:
    print(f"  Target classes distribution:")
    target_class_counts = df['target__class'].value_counts()
    for cls, count in target_class_counts.items():
        print(f"    {cls}: {count:,} ({count/len(df)*100:.2f}%)")

if 'target__gene' in df.columns:
    unique_genes = df['target__gene'].nunique()
    print(f"\n  Unique target genes: {unique_genes:,}")
    print(f"  Top 10 most studied genes:")
    top_genes = df['target__gene'].value_counts().head(10)
    for gene, count in top_genes.items():
        print(f"    {gene}: {count:,} assays")

if 'target__is_mutant' in df.columns:
    mutant_count = df['target__is_mutant'].sum()
    wildtype_count = (~df['target__is_mutant']).sum()
    print(f"\n  Mutant targets: {mutant_count:,} ({mutant_count/len(df)*100:.2f}%)")
    print(f"  Wild-type targets: {wildtype_count:,} ({wildtype_count/len(df)*100:.2f}%)")

# 5.2 Compound Analysis
print("\n5.2 Compound Analysis:")
print("-" * 80)
if 'compound_id' in df.columns:
    unique_compounds = df['compound_id'].nunique()
    print(f"  Unique compounds tested: {unique_compounds:,}")

    print(f"  Top 10 most tested compounds:")
    top_compounds = df['compound_id'].value_counts().head(10)
    for i, (comp_id, count) in enumerate(top_compounds.items(), 1):
        comp_name = df[df['compound_id'] == comp_id]['compound__name'].iloc[0]
        print(f"    {i}. {comp_name} ({comp_id}): {count:,} assays")

if 'compound__drugbank_id' in df.columns:
    drugbank_count = df['compound__drugbank_id'].notna().sum()
    print(f"\n  Compounds with DrugBank ID: {drugbank_count:,} ({drugbank_count/len(df)*100:.2f}%)")

# 5.3 Assay Analysis
print("\n5.3 Assay Analysis:")
print("-" * 80)
if 'assay__technology' in df.columns:
    print(f"  Assay technologies:")
    tech_counts = df['assay__technology'].value_counts()
    for tech, count in tech_counts.items():
        print(f"    {tech}: {count:,} ({count/len(df)*100:.2f}%)")

if 'mode' in df.columns:
    print(f"\n  Interaction modes:")
    mode_counts = df['mode'].value_counts()
    for mode, count in mode_counts.items():
        print(f"    {mode}: {count:,} ({count/len(df)*100:.2f}%)")

if 'mechanism' in df.columns:
    print(f"\n  Mechanisms:")
    mech_counts = df['mechanism'].value_counts()
    for mech, count in mech_counts.items():
        print(f"    {mech}: {count:,} ({count/len(df)*100:.2f}%)")

# 5.4 Activity Analysis
print("\n5.4 Activity Analysis:")
print("-" * 80)
if 'outcome_is_active' in df.columns:
    active_count = df['outcome_is_active'].sum()
    inactive_count = (~df['outcome_is_active']).sum()
    print(f"  Active compounds: {active_count:,} ({active_count/len(df)*100:.2f}%)")
    print(f"  Inactive compounds: {inactive_count:,} ({inactive_count/len(df)*100:.2f}%)")

if 'is_quantified' in df.columns:
    quantified = df['is_quantified'].sum()
    print(f"\n  Quantified measurements: {quantified:,} ({quantified/len(df)*100:.2f}%)")

if 'outcome_potency_pxc50' in df.columns:
    print(f"\n  Potency (pXC50) statistics:")
    potency_data = df['outcome_potency_pxc50'].dropna()
    print(f"    Mean: {potency_data.mean():.2f}")
    print(f"    Median: {potency_data.median():.2f}")
    print(f"    Std Dev: {potency_data.std():.2f}")
    print(f"    Min: {potency_data.min():.2f}")
    print(f"    Max: {potency_data.max():.2f}")
    print(f"    Range: {potency_data.max() - potency_data.min():.2f}")

if 'outcome_max_activity' in df.columns:
    print(f"\n  Max activity statistics:")
    max_act_data = df['outcome_max_activity'].dropna()
    print(f"    Mean: {max_act_data.mean():.2f}")
    print(f"    Median: {max_act_data.median():.2f}")
    print(f"    Std Dev: {max_act_data.std():.2f}")
    print(f"    Min: {max_act_data.min():.2f}")
    print(f"    Max: {max_act_data.max():.2f}")

# 5.5 Slope and Curve Parameters
print("\n5.5 Dose-Response Curve Parameters:")
print("-" * 80)
curve_params = ['slope', 'asymp_min', 'asymp_max']
for param in curve_params:
    if param in df.columns:
        param_data = df[param].dropna()
        print(f"\n  {param}:")
        print(f"    Mean: {param_data.mean():.3f}")
        print(f"    Median: {param_data.median():.3f}")
        print(f"    Std Dev: {param_data.std():.3f}")
        print(f"    Min: {param_data.min():.3f}")
        print(f"    Max: {param_data.max():.3f}")
        print(f"    Q1: {param_data.quantile(0.25):.3f}")
        print(f"    Q3: {param_data.quantile(0.75):.3f}")

# 5.6 Release Analysis
print("\n5.6 Data Release Distribution:")
print("-" * 80)
if 'release' in df.columns:
    release_counts = df['release'].value_counts().sort_index()
    print(f"  Number of releases: {len(release_counts)}")
    print(f"  Release distribution:")
    for release, count in release_counts.items():
        print(f"    Release {release}: {count:,} ({count/len(df)*100:.2f}%)")

# 5.7 Progressed Compounds
print("\n5.7 Drug Progression Status:")
print("-" * 80)
if 'progressed' in df.columns:
    progressed_count = df['progressed'].sum()
    print(f"  Progressed compounds: {progressed_count:,} ({progressed_count/len(df)*100:.2f}%)")
    print(f"  Not progressed: {len(df) - progressed_count:,} ({(len(df) - progressed_count)/len(df)*100:.2f}%)")

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6. CORRELATION ANALYSIS (Top Correlations)")
print("="*80)

if len(numerical_cols) > 1:
    corr_matrix = df[numerical_cols].corr()

    # Get upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find correlations above threshold
    correlations = []
    for column in upper_tri.columns:
        for index in upper_tri.index:
            corr_val = upper_tri.loc[index, column]
            if pd.notna(corr_val) and abs(corr_val) > 0.3:
                correlations.append((index, column, corr_val))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\nFound {len(correlations)} correlations with |r| > 0.3:")
    print("-" * 80)
    for var1, var2, corr in correlations[:20]:
        print(f"  {var1} ↔ {var2}: {corr:.3f}")

# ============================================================================
# 7. DATA QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("7. DATA QUALITY ASSESSMENT")
print("="*80)

print("\n7.1 Missing Data Summary:")
print("-" * 80)
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing_data / len(df) * 100).sort_values(ascending=False)

print(f"  Columns with missing data:")
for col, count in missing_data[missing_data > 0].items():
    pct = missing_pct[col]
    print(f"    {col}: {count:,} ({pct:.2f}%)")

if missing_data.sum() == 0:
    print("  ✓ No missing data detected!")

print("\n7.2 Duplicate Rows:")
print("-" * 80)
duplicate_count = df.duplicated().sum()
print(f"  Duplicate rows: {duplicate_count:,} ({duplicate_count/len(df)*100:.2f}%)")

print("\n7.3 Data Completeness:")
print("-" * 80)
total_cells = df.shape[0] * df.shape[1]
non_null_cells = df.count().sum()
completeness = non_null_cells / total_cells * 100
print(f"  Total cells: {total_cells:,}")
print(f"  Non-null cells: {non_null_cells:,}")
print(f"  Data completeness: {completeness:.2f}%")

# ============================================================================
# 8. SUMMARY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("8. KEY INSIGHTS & SUMMARY")
print("="*80)

print(f"""
Dataset Characteristics:
  • Total drug-target interactions: {len(df):,}
  • Unique compounds: {df['compound_id'].nunique() if 'compound_id' in df.columns else 'N/A'}
  • Unique targets: {df['target_id'].nunique() if 'target_id' in df.columns else 'N/A'}
  • Unique assays: {df['assay_id'].nunique() if 'assay_id' in df.columns else 'N/A'}
  • Data completeness: {completeness:.2f}%

Target Coverage:
  • Target classes: {df['target__class'].nunique() if 'target__class' in df.columns else 'N/A'}
  • Unique genes: {df['target__gene'].nunique() if 'target__gene' in df.columns else 'N/A'}
  • Mutant variants: {df['target__is_mutant'].sum() if 'target__is_mutant' in df.columns else 'N/A'}

Compound Characteristics:
  • DrugBank annotated: {df['compound__drugbank_id'].notna().sum() if 'compound__drugbank_id' in df.columns else 'N/A'}
  • Progressed compounds: {df['progressed'].sum() if 'progressed' in df.columns else 'N/A'}
  • Active hit rate: {df['outcome_is_active'].sum()/len(df)*100:.2f}% if 'outcome_is_active' in df.columns else 'N/A'

Assay Quality:
  • Quantified measurements: {df['is_quantified'].sum()/len(df)*100:.2f}% if 'is_quantified' in df.columns else 'N/A'
  • Average potency (pXC50): {df['outcome_potency_pxc50'].mean():.2f} if 'outcome_potency_pxc50' in df.columns else 'N/A'
  • Technology platforms: {df['assay__technology'].nunique() if 'assay__technology' in df.columns else 'N/A'}
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
