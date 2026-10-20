"""
Synthetic Healthcare Insurance Dataset Generator
Generates a realistic synthetic insurance dataset labeled as 'Synthetic/Demo Dataset'
for actuarial cost and risk modeling.
"""

import os
import numpy as np
import pandas as pd

def generate_insurance_dataset(n_samples: int = 6000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic insurance dataset with realistic demographic, lifestyle,
    and actuarial risk factors.
    """
    np.random.seed(random_state)

    # 1. Age: 18 to 65 uniformly distributed with realistic slight skew
    ages = np.random.randint(18, 66, size=n_samples)

    # 2. Sex: roughly 50.5% female, 49.5% male
    sexes = np.random.choice(['female', 'male'], size=n_samples, p=[0.505, 0.495])

    # 3. BMI: Normal distribution around 30.6 (mean US adult BMI) with std 6.1, clipped between 15.0 and 53.0
    bmis = np.random.normal(loc=30.6, scale=6.1, size=n_samples)
    bmis = np.clip(bmis, 15.0, 53.0)
    bmis = np.round(bmis, 1)

    # 4. Children (dependents): 0 to 5, skewed towards 0 and 1
    child_probs = [0.42, 0.24, 0.18, 0.11, 0.035, 0.015]
    children = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=child_probs)

    # 5. Smoker: ~20% smokers (slightly higher among males in general demographics)
    smoker_prob = np.where(sexes == 'male', 0.22, 0.18)
    smoker_rand = np.random.rand(n_samples)
    smokers = np.where(smoker_rand < smoker_prob, 'yes', 'no')

    # 6. Region: 4 geographic regions
    regions = np.random.choice(['southeast', 'southwest', 'northwest', 'northeast'], size=n_samples, p=[0.27, 0.25, 0.24, 0.24])

    # 7. Actuarial Healthcare Charges Simulation:
    # Base fee + Age slope + BMI slope + Dependents fee + Regional modifier + Smoker penalty + Interaction
    base_charge = 2800.0
    age_component = ages * 255.0
    bmi_component = np.maximum(0, bmis - 25.0) * 320.0
    children_component = children * 460.0

    # Regional cost of care adjustment
    region_factors = {
        'southeast': 1.05,
        'northeast': 1.03,
        'northwest': 0.98,
        'southwest': 0.96
    }
    reg_mult = np.array([region_factors[r] for r in regions])

    # Smoker impact: Non-smokers have modest costs; smokers have extreme cost escalations
    # Especially if BMI >= 30 (known medical/insurance compound risk threshold)
    smoker_base_penalty = np.where(smokers == 'yes', 14500.0, 0.0)
    obese_smoker_penalty = np.where((smokers == 'yes') & (bmis >= 30.0), 16000.0 + (bmis - 30.0) * 450.0, 0.0)

    # Idiosyncratic health variance (log-normal distributed random healthcare incidents)
    noise = np.random.normal(0, 750.0, size=n_samples)
    incidents = np.random.exponential(scale=600.0, size=n_samples)

    total_charges = (base_charge + age_component + bmi_component + children_component + smoker_base_penalty + obese_smoker_penalty + noise + incidents) * reg_mult
    total_charges = np.clip(total_charges, 1120.0, 65000.0)
    total_charges = np.round(total_charges, 2)

    df = pd.DataFrame({
        'age': ages,
        'sex': sexes,
        'bmi': bmis,
        'children': children,
        'smoker': smokers,
        'region': regions,
        'charges': total_charges
    })

    return df

def save_raw_dataset(output_path: str | None = None, n_samples: int = 6000) -> str:
    """Generate and save the raw synthetic dataset."""
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_path = os.path.join(base_dir, 'data', 'raw', 'insurance_dataset.csv')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_insurance_dataset(n_samples=n_samples)
    df.to_csv(output_path, index=False)
    print(f"[Dataset Generator] Successfully generated {len(df)} records at: {output_path}")
    return output_path

if __name__ == '__main__':
    save_raw_dataset()
