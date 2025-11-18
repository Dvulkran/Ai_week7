## Week 7 – Ethical AI & Bias Auditing

### Overview
Week 7 focuses on **Ethical AI** and **Bias Detection** in machine learning systems. This week explores the critical importance of fairness, transparency, and accountability in AI applications, with a practical deep-dive into auditing the COMPAS Recidivism Dataset for racial bias using IBM's AI Fairness 360 toolkit. The work demonstrates how algorithmic systems can perpetuate and amplify societal biases, and provides methodologies for detecting, measuring, and mitigating such biases.

---

## Part 1: Theoretical Understanding

### Key Concepts Covered

#### Ethical AI Principles
- **Fairness**: Ensuring equitable treatment across different demographic groups
- **Transparency**: Making AI decision-making processes understandable
- **Accountability**: Establishing responsibility for AI system outcomes
- **Privacy**: Protecting individual data and maintaining confidentiality
- **Non-discrimination**: Preventing bias based on protected attributes

#### Types of Bias in AI
1. **Historical Bias**: Bias present in training data reflecting past discrimination
2. **Representation Bias**: Underrepresentation or misrepresentation of certain groups
3. **Measurement Bias**: Inaccurate or incomplete measurement of outcomes
4. **Aggregation Bias**: Applying models uniformly across diverse populations
5. **Evaluation Bias**: Using inappropriate metrics or benchmarks
6. **Deployment Bias**: Misalignment between training context and real-world use

#### Fairness Metrics
- **Disparate Impact**: Ratio of favorable outcomes between groups (should be ~1.0)
- **Statistical Parity**: Equal probability of positive outcomes across groups
- **Equalized Odds**: Equal true positive and false positive rates across groups
- **Calibration**: Similar predicted probabilities for similar risk levels
- **Individual Fairness**: Similar individuals receive similar predictions

#### Protected Attributes
Attributes that should not be used for discriminatory purposes:
- Race/Ethnicity
- Gender/Sex
- Age
- Religion
- National Origin
- Disability Status
- Sexual Orientation

---

## Part 2: Guidelines for Ethical AI Use

### Best Practices

#### Development Phase
- **Diverse Teams**: Include diverse perspectives in AI development
- **Bias Testing**: Regular audits during model development
- **Documentation**: Maintain detailed records of data sources and decisions
- **Stakeholder Engagement**: Involve affected communities in design

#### Deployment Phase
- **Transparency**: Clearly communicate how AI systems work
- **Human Oversight**: Maintain human review for high-stakes decisions
- **Monitoring**: Continuous monitoring for bias and performance degradation
- **Feedback Loops**: Mechanisms for users to report issues

#### Post-Deployment
- **Regular Audits**: Periodic bias assessments and fairness evaluations
- **Impact Assessment**: Evaluate real-world effects on different groups
- **Remediation Plans**: Procedures for addressing identified biases
- **Public Reporting**: Transparent reporting of system performance

### Ethical Frameworks
- **Utilitarian**: Maximize overall benefit while minimizing harm
- **Deontological**: Follow moral rules and duties regardless of consequences
- **Virtue Ethics**: Focus on character and moral excellence
- **Rights-Based**: Protect fundamental human rights

---

## Part 3: Practical Audit - COMPAS Recidivism Dataset

### Objective
Conduct a comprehensive bias audit of the COMPAS (Correctional Offender Management Profiling for Alternative Sanctions) recidivism risk assessment tool to identify and quantify racial disparities in risk score predictions.

### Dataset Overview
- **Source**: ProPublica COMPAS Analysis Dataset
- **Records**: 7,214 individuals
- **Time Period**: 2013-2014
- **Geographic Scope**: Broward County, Florida
- **Purpose**: Predict recidivism risk within two years

#### Key Features
- **Demographics**: Age, Sex, Race
- **Criminal History**: Juvenile felony/misdemeanor counts, prior convictions
- **Charge Information**: Charge degree (Felony/Misdemeanor)
- **Risk Scores**: COMPAS decile score (1-10, higher = higher risk)
- **Outcome**: Two-year recidivism (binary: 0 = no recidivism, 1 = recidivism)

#### Racial Distribution
- African-American: ~51%
- Caucasian: ~34%
- Hispanic: ~10%
- Other: ~5%

### Methodology

#### Tools & Libraries
- **IBM AI Fairness 360 (AIF360)**: Comprehensive fairness toolkit
- **Fairlearn**: Microsoft's fairness assessment library
- **Scikit-learn**: Machine learning metrics and utilities
- **Pandas/NumPy**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Visualization

#### Analysis Steps

1. **Data Loading & Exploration**
   - Load COMPAS dataset from ProPublica repository
   - Exploratory data analysis (EDA)
   - Feature selection and cleaning
   - Missing value handling

2. **Data Preprocessing**
   - Binary encoding of categorical variables
   - Risk score thresholding (high risk: decile ≥ 5)
   - Focus on African-American vs. Caucasian comparison

3. **Fairness Metrics Calculation**
   - **Disparate Impact**: Ratio of favorable outcomes
   - **Statistical Parity Difference**: Difference in positive outcome rates
   - **Mean Difference**: Average outcome difference between groups

4. **Error Rate Analysis**
   - **False Positive Rate (FPR)**: Incorrectly labeled as high risk
   - **False Negative Rate (FNR)**: Incorrectly labeled as low risk
   - **True Positive Rate (TPR)**: Correctly identified high risk
   - **True Negative Rate (TNR)**: Correctly identified low risk

5. **Visualization**
   - Risk score distribution by race
   - Recidivism rates by race
   - Error rate disparities
   - Confusion matrices by race

### Key Findings

#### Fairness Metrics Results
- **Disparate Impact**: Typically < 0.8 (indicating bias)
  - Interpretation: African-Americans receive fewer favorable outcomes
  - Threshold: Should be close to 1.0 for fairness

- **Statistical Parity Difference**: Typically < -0.1 (indicating bias)
  - Interpretation: Significant difference in favorable outcome rates
  - Threshold: Should be close to 0.0 for fairness

#### Error Rate Disparities
- **False Positive Rate Disparity**: African-Americans often have 5-10% higher FPR
  - Meaning: More likely to be incorrectly classified as high risk
  - Impact: Unjustified restrictions, harsher treatment

- **False Negative Rate Disparity**: Caucasians often have higher FNR
  - Meaning: More likely to be incorrectly classified as low risk
  - Impact: Missed interventions, public safety concerns

#### Risk Score Patterns
- **Average Risk Scores**: Higher for African-Americans
- **High Risk Classification**: Disproportionate rates by race
- **Actual Recidivism**: Similar rates across groups, but different predictions

### Visualizations Generated

1. **`compas_bias_overview.png`**
   - Risk score distribution by race
   - Actual recidivism rates
   - Predicted high risk rates
   - Average risk scores

2. **`compas_error_rates.png`**
   - False positive rate comparison
   - False negative rate comparison
   - Overall accuracy by race
   - Error rate side-by-side comparison

3. **`compas_confusion_matrices.png`**
   - Confusion matrices for each racial group
   - Visual representation of prediction errors

### Bias Remediation Recommendations

#### Algorithmic Interventions
1. **Fairness-Aware Algorithms**
   - Implement algorithms that explicitly optimize for fairness
   - Use techniques like reweighing, threshold optimization
   - Apply disparate impact remover preprocessing

2. **Threshold Adjustment**
   - Use different risk score thresholds for different groups
   - Optimize thresholds to equalize error rates
   - Implement calibrated risk assessments

3. **Feature Engineering**
   - Remove race-correlated proxy features
   - Add contextual socioeconomic features
   - Include rehabilitation and support access data
   - Weight positive behavioral indicators more heavily

#### Data Collection Improvements
1. **Representative Sampling**: Ensure balanced representation across demographics
2. **Granular Data**: Collect more detailed socioeconomic and contextual information
3. **Historical Bias Tracking**: Include indicators of past discrimination
4. **Quality Assurance**: Regular audits for data completeness and accuracy

#### Procedural Safeguards
1. **Human Oversight**: Require human review for high-stakes decisions
2. **Justification Requirements**: Mandate explanations for risk score overrides
3. **Regular Audits**: Quarterly/annual bias assessments
4. **Transparency**: Clear communication of system limitations

#### Monitoring & Evaluation
1. **Continuous Tracking**: Monitor fairness metrics over time
2. **A/B Testing**: Test bias mitigation strategies
3. **Feedback Mechanisms**: Collect input from affected populations
4. **Public Reporting**: Annual fairness reports

#### Policy Recommendations
1. **Usage Limitations**: Restrict algorithmic risk assessment in sentencing
2. **Independent Audits**: Require third-party evaluations before deployment
3. **Certification Standards**: Establish fairness certification requirements
4. **Appeals Process**: Create mechanisms for contesting risk scores

---

## Technical Requirements

### Dependencies

```python
aif360>=0.5.0          # IBM AI Fairness 360 toolkit
fairlearn>=0.8.0        # Microsoft Fairlearn
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
```

### Installation

```bash
# Install all required packages
pip install aif360 fairlearn pandas numpy matplotlib seaborn scikit-learn
```

### Platform
- **Recommended**: Google Colab (for easy setup and package installation)
- **Alternative**: Local Jupyter environment with Python 3.8+

---

## Usage Instructions

### Running the Bias Audit

1. **Open the Notebook**
   - Open `wk7_task3.ipynb` in Google Colab or Jupyter

2. **Install Dependencies**
   - The notebook automatically installs required packages
   - Run the first cell to install: `aif360`, `fairlearn`, and data science libraries

3. **Execute Analysis**
   - Run all cells sequentially
   - The notebook will:
     - Load and explore the COMPAS dataset
     - Preprocess data for analysis
     - Calculate fairness metrics using AIF360
     - Generate comprehensive visualizations
     - Produce detailed bias audit report

4. **Review Results**
   - Check generated visualizations (PNG files)
   - Review fairness metrics summary
   - Examine error rate disparities
   - Read remediation recommendations

### Expected Outputs

1. **Console Output**
   - Dataset statistics
   - Fairness metrics (Disparate Impact, Statistical Parity, etc.)
   - Error rate analysis by race
   - Comprehensive summary report

2. **Visualization Files**
   - `compas_bias_overview.png` - Overall bias indicators
   - `compas_error_rates.png` - Error rate analysis
   - `compas_confusion_matrices.png` - Confusion matrices by race

3. **Summary Report**
   - 300+ word analysis of findings
   - Specific bias indicators
   - Remediation recommendations

---

## Key Learnings

### Bias Detection
- How to identify bias in real-world datasets
- Understanding different types of fairness metrics
- Interpreting disparate impact and statistical parity
- Analyzing error rate disparities across groups

### Fairness Toolkits
- Using IBM AI Fairness 360 for comprehensive analysis
- Understanding protected attributes and privileged groups
- Calculating fairness metrics programmatically
- Visualizing bias patterns

### Real-World Implications
- Impact of biased algorithms on marginalized communities
- Legal and ethical considerations in risk assessment
- Importance of transparency and accountability
- Need for continuous monitoring and remediation

### Remediation Strategies
- Algorithmic approaches to bias mitigation
- Data collection and preprocessing improvements
- Procedural safeguards and human oversight
- Policy and regulatory considerations

---

## Files Structure

```
week 7/
├── README.md                              # This file
├── wk7_task3.ipynb                        # Main bias audit notebook
├── guideline for ethical AI use.pdf       # Ethical AI guidelines
├── part1_Theoretical Understanding.pdf    # Theoretical foundations
├── Part3_Audit of a Dataset for Bias.pdf  # Bias audit methodology
└── Generated Files/                       # (Created when running notebook)
    ├── compas_bias_overview.png
    ├── compas_error_rates.png
    └── compas_confusion_matrices.png
```

---

## Important Notes

### Ethical Considerations
- This analysis is for educational purposes to understand bias in AI systems
- The COMPAS dataset reflects real-world bias that has had serious consequences
- Results may vary slightly between runs due to data sampling
- The analysis focuses on African-American vs. Caucasian comparison for clarity

### Limitations
- Analysis uses binary racial categories for simplicity
- Synthetic data fallback if ProPublica URL is unavailable
- Focus on two-year recidivism window
- Does not address all forms of bias (e.g., intersectionality)

### Real-World Context
- COMPAS has been used in criminal justice decisions
- ProPublica's 2016 investigation revealed significant racial bias
- Multiple jurisdictions have restricted or banned COMPAS use
- Ongoing debate about algorithmic risk assessment in justice systems

---

## References

### Academic Papers
- Angwin, J., et al. (2016). "Machine Bias: There's software used across the country to predict future criminals. And it's biased against blacks." ProPublica
- Chouldechova, A. (2017). "Fair prediction with disparate impact: A study of bias in recidivism prediction instruments." Big Data
- Dieterich, W., et al. (2016). "COMPAS Risk Scales: Demonstrating Accuracy Equity and Predictive Parity." Northpointe Inc.

### Toolkits & Frameworks
- **IBM AI Fairness 360**: https://aif360.mybluemix.net/
- **Fairlearn**: https://fairlearn.org/
- **COMPAS Dataset**: https://github.com/propublica/compas-analysis

### Ethical AI Guidelines
- IEEE Ethically Aligned Design
- EU AI Act
- ACM Code of Ethics
- Partnership on AI Tenets

---

## Future Enhancements

### Analysis Improvements
- Expand to include intersectional analysis (race × gender × age)
- Include additional fairness metrics (equalized odds, calibration)
- Implement bias mitigation algorithms (reweighing, prejudice remover)
- Compare multiple risk assessment tools

### Technical Enhancements
- Real-time bias monitoring dashboard
- Automated fairness report generation
- Integration with model training pipelines
- Bias detection in streaming data

### Educational Extensions
- Interactive bias visualization tools
- Case studies of other biased systems
- Hands-on bias mitigation exercises
- Policy analysis and recommendations

---

## Conclusion

Week 7 provides critical insights into the ethical dimensions of AI systems. Through the practical audit of the COMPAS dataset, we learn that:

1. **Bias is pervasive**: Even well-intentioned systems can perpetuate discrimination
2. **Measurement matters**: Proper metrics are essential for detecting bias
3. **Transparency is crucial**: Understanding how systems work enables accountability
4. **Remediation is possible**: Multiple strategies exist to address bias
5. **Ongoing vigilance**: Continuous monitoring is necessary to maintain fairness

The skills developed in this week are essential for anyone working with AI systems that impact human lives, particularly in high-stakes domains like criminal justice, healthcare, hiring, and lending.

---

**Week 7 Complete**: Ethical AI principles and bias auditing successfully implemented! ⚖️🤖

