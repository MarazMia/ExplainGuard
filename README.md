# ExplainGuard: Post-Hoc Explanation Auditing & Adversarial Defense Framework

ExplainGuard is a comprehensive security audit verification framework designed to detect, evaluate, and defend against adversarial manipulations of post-hoc machine learning explanations (such as SHAP attributions). 

Modern ML auditing standards often rely on black-box verification checks to ensure that explanations submitted by auditors have been faithful. ExplainGuard implements and tests three major threat models against a multi-pillar verification pipeline:
1. **OOD Scaffolding Attack** (Slack et al., AIES 2020)
2. **Output Shuffling Attack** (Yuan & Dasgupta, AAAI 2024)
3. **Stealthily Biased Sampling Attack** (Laberge et al., ICLR 2023)

---

## Project Structure

```text
ExplainGuard/
│
├── Audit/
│   ├── audit_attack_BiasedSampling.py   # Laberge et al. (ICLR '23) Algorithm 3 Implementation
│   └── ...                             # Other attack harness scripts
│
├── Data/
│   └── Auditor_End/
│       ├── compas/                     # COMPAS dataset split artifacts
│       └── Submissions/                # Serialized .pkl audit submission packages
│   └── Owner_End/                      # Model Owner End data
│   └── Dev_End/                        # Model Developer End data
│
├── Model/
│   ├── blackbox_endpoint.py            # Standardized model wrapper endpoints
│   └── original_compas_xgboost_model.json
│
├── Pillars_Check/
│   └── verifier_pillar1.py             # Pillar I Verification & Security Checks
│   └── ...                             # Other two pillars
│
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation
```


### Step 1.
Clone the repository and navigate to the project root:
<i> cd ExplainGuard </i>

### Step 2.
#### Create and activate a virtual environment:
<i>
python -m venv .venv <br>
# On Windows: <br>
.venv\Scripts\activate <br>
# On macOS/Linux: <br>
source .venv/bin/activate <br>
</i>

### Step 3.
#### Install the required dependencies:

<i> pip install -r requirements.txt </i>


### Step 4.
#### Develop the original model:

<i> python Model/developing_clean_model.py </i>

### Step 5.
#### Running the Audit & Verification Pipeline
Sub-Step 1: Execute the Biased Sampling Attack (or the other variant. Can also check the result for the clean audit result using the same steps.)

<i> python Audit/audit_attack_BiasedSampling.py </i>


Sub-Step 2: Run the Pillar I, II, and III Verification Harness

<i> python Pillars_Check/verifier_pillar1.py </i> <br>
<i> python Pillars_Check/verifier_pillar2.py </i> <br>
<i> python Pillars_Check/verifier_pillar3.py </i> <br>


