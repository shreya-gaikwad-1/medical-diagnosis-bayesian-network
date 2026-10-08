# Medical Diagnosis under Uncertainty using Bayesian Networks and Decision Theory

Advanced AI mini project. A clinical decision-support system that estimates a patient's probability of lung cancer from risk factors and findings, then recommends whether to treat using expected utility.

## Syllabus Topics Covered

| Unit | Topic | Where in the code |
|------|-------|-------------------|
| 1 | Bayesian Networks, conditional independence | `BN` dictionary in `medical_diagnosis_bn.py` |
| 2 | Exact inference: Variable Elimination | `variable_elimination()` |
| 3 | Approximate inference: Sampling | `likelihood_weighting()` |
| 5 | Expected utility, Value of Information | `best_action()`, `value_of_information()` |

## Bayesian Network

```
Smoker ──┐
         ├──> Cancer ──┬──> XRay
Pollution┘             └──> Dyspnea
```

All variables are binary (0 = No, 1 = Yes). The probabilities are illustrative values chosen for the demo, not real patient data.

## Features

- Updates P(Cancer) as evidence (smoking, pollution, X-ray, dyspnea) is added
- Compares exact inference with Likelihood Weighting and shows convergence as samples increase
- Recommends Treat or Don't treat by maximizing expected utility, with adjustable payoffs
- Calculates the Value of Information of ordering an extra test
- Interactive Streamlit dashboard with network diagram and probability tables

## Tools Used

Python, Streamlit, pandas, Graphviz (rendered by Streamlit)

## Project Structure

```
app.py                    # Streamlit dashboard
medical_diagnosis_bn.py   # Inference and decision engine (pure Python)
requirements.txt          # Dependencies
README.md
```

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

To run only the command-line version:

```bash
python medical_diagnosis_bn.py
```

## Sample Output

| Scenario | Exact P(Cancer) | Likelihood Weighting |
|----------|-----------------|----------------------|
| Smoker | 0.0340 | 0.0364 |
| Smoker + Dyspnea | 0.1026 | 0.1014 |
| Smoker + Dyspnea + positive X-Ray | 0.6731 | 0.6891 |
| Non-smoker + negative X-Ray | 0.0005 | 0.0006 |

Recommended action: Don't treat for a smoker with no symptoms; Treat once Dyspnea is observed.

## Limitations and Future Work

- Probabilities are hand-specified; they could be learned from a real dataset (parameter estimation, Unit 3)
- Add Gibbs sampling for comparison with Likelihood Weighting
- Add hidden variables and learn them with EM (Unit 4)

## Disclaimer

This is an academic project and must not be used for real medical decisions.
