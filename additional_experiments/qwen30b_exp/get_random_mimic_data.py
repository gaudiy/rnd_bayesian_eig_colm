"""
This script processes the `mimic-iv-ed-2.2` datasets. 

"""

import pandas as pd 
import json 
import yaml 

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
if __name__=="__main__":

    # Configurations
    cfg = load_config('config.yaml')
    mimic_cfg = cfg['mimic']
    mimic_ed_folder = mimic_cfg['data_source_folder']
    data_cfg = cfg['data']


    n_diseases_num = mimic_cfg['n_diseases_num'] 
    n_top_diseases_num = mimic_cfg['n_top_diseases_num']
    n_patients_num = mimic_cfg['n_patients_num']
    random_select_seed = mimic_cfg['random_select_seed']

    # load data
    df = pd.read_csv(f"{mimic_ed_folder}/diagnosis.csv")

    # We do this NOW so we only count diseases that actually caused the admission.
    df = df[df['seq_num'] == 1]
    len(df), len(set(df.stay_id)), len(set(df.subject_id))

    unique_raw_icd_titles = set(df.icd_title)
    print(f'There are {len(unique_raw_icd_titles)} unique raw icd titlees')

    mask_icd10_junk = (df['icd_version'] == 10) & (df['icd_code'].str.contains(r'^[RZVYWX]', regex=True))

    # --- PRE-PROCESSING ---

    # Ensure codes are strings
    df['icd_code'] = df['icd_code'].astype(str)
    df['icd_title'] = df['icd_title'].astype(str)

    # Standardize titles (Merge "HYPERTENSION" and "Hypertension")
    df['clean_title'] = df['icd_title'].str.title().str.strip()

    # --- FILTERING LOGIC ---

    # Create a mask for rows we want to KEEP
    keep_mask = pd.Series(True, index=df.index)

    # 1. Filter ICD-10 Junk (Symptoms R*, Status Z*, Accidents V-Y*)
    mask_icd10_junk = (df['icd_version'] == 10) & (df['icd_code'].str.contains(r'^[RZVYWX]', regex=True))
    keep_mask = keep_mask & ~mask_icd10_junk

    # 2. Filter ICD-9 Junk (External E*, Status V*, General Symptoms 78-79*)
    mask_icd9_junk = (df['icd_version'] == 9) & (
        df['icd_code'].str.contains(r'^[EV]', regex=True) | 
        df['icd_code'].str.contains(r'^7[89]', regex=True)
    )
    keep_mask = keep_mask & ~mask_icd9_junk

    # --- EXECUTE ---

    # Apply the filters
    df_diseases = df[keep_mask]

    unique_diseases = list(set(df_diseases['clean_title']))

    print(f'There are a total of {len(unique_diseases)} unique diseases!')
    print(f'We have a total of {len(df_diseases)} unique visits/statys')

    with open(data_cfg['disease_mapping_path'], 'r') as file:
        TARGET_MAP = json.load(file)

    disease_freq_dict = df_diseases['clean_title'].value_counts().to_dict()
    total = 0
    for k, v in disease_freq_dict.items():
        if k in TARGET_MAP:
            total += v 
    print(f'{total/len(df_diseases)}')

    # 1. Map and Clean
    df['target_label'] = df['clean_title'].map(TARGET_MAP)
    df_clean = df.dropna(subset=['target_label']).copy()
    n_df_clean_before_filtering = len(df_clean)
    print(f'After excluding data whose ICD titles are not in the top 500, we have {len(df_clean)} visits!')

    print(f"Now filtering for the Top {n_top_diseases_num} Primary Diseases.")

    # 3. Find the Top N most common *Primary* labels
    top_n_labels = df_clean['target_label'].value_counts().head(n_top_diseases_num).index.tolist()

    with open(data_cfg['disease_list_path'], "w", encoding="utf-8") as f:
        for label in top_n_labels:
            f.write(f"{label}\n")

    df_clean = df_clean[df_clean['target_label'].isin(top_n_labels)]
    print(
        f"""
    After we use only top {n_top_diseases_num} target labels,
    we have {len(df_clean)} visits,
    accounting for {len(df_clean) / n_df_clean_before_filtering:.2%}
    of all the visits of the top 500 clean titles or 238 target labels!
    """
    )

    # --- 4. MERGE & FILTER FOR COMPLETENESS ---

    print("Loading additional data...")
    df_triage = pd.read_csv(f"{mimic_ed_folder}/triage.csv")
    df_edstays = pd.read_csv(f"{mimic_ed_folder}/edstays.csv")

    # 1. Merge Diagnosis with Triage immediately
    # We do this BEFORE sampling to check who has valid data
    full_pool = pd.merge(df_clean, df_triage, on='stay_id', how='left')

    required_cols = [
        'temperature', 'heartrate', 'resprate', 'o2sat',
        'sbp', 'dbp', 'pain', 'acuity',
        'chiefcomplaint'
    ]

    numeric_cols = [c for c in required_cols if c != 'chiefcomplaint']

    valid_pool = full_pool[
        # chief complaint must exist
        full_pool['chiefcomplaint'].notna()
        &
        # all numeric cols must be numeric and not NA
        full_pool[numeric_cols]
            .apply(pd.to_numeric, errors='coerce')
            .notna()
            .all(axis=1)
    ].copy()

    # SHOULD CHECK THESE COLS' TYPE! SOME ARE NOT THE CORRECT TYPE. E.G., 
    # PAIN SHOULD BE NUM NOT STR

    print(f"Original pool size: {len(df_clean)}")
    print(f"Visits/Stays with valid vitals: {len(valid_pool)}")
    # 3. Sample 10 Random Patients from the valid pool
    # Now we are guaranteed that all 10 have usable data
    sampled_df = valid_pool.sample(n=n_patients_num, random_state=random_select_seed).copy()

    # 4. Add Demographics (from edstays.csv)
    # We want: Gender, Race, Arrival Transport
    cols_edstays = ['stay_id', 'gender', 'race', 'arrival_transport']
    sampled_df = pd.merge(sampled_df, df_edstays[cols_edstays], on='stay_id', how='left')
    print(sampled_df.columns)
    print(f"After sampling, we have {len(sampled_df.target_label.unique())} target labels.")
    sampled_df = sampled_df.dropna()
    sampled_df.to_csv(data_cfg['data_path'], index=False)