# Feature combinations and targets
feature_combs = ["Sleep", "Cov",
                 "Brain", "CT", "SA", "Subcor",
                 "Sleep_Cov", "Sleep_Cov_Brain", "Sleep_Cov_CT", "Sleep_Cov_SA", "Sleep_Cov_Subcor",
                 "Sleep_Brain", "Sleep_CT", "Sleep_SA", "Sleep_Subcor",
                 "Cov_Brain", "Cov_CT", "Cov_SA", "Cov_Subcor",
                 "Sleep_Shuffle_Cov", "Sleep_Cov_Subcor_Shuffle", "Sleep_Shuffle_Cov_Subcor",
                 "Sleep_Shuffle_Subcor", "Cov_Brain_Shuffle", "Cov_Subcor_Shuffle"]
targets = ["Stroop", "Memory"]

# Path for the DAG file
dag_file = "dag_SVM-linear_shap.dag"

with open(dag_file, 'w') as f:
    for target in targets:
        for feature_comb in feature_combs:
            # Define SVM-linear job
            f.write(f"JOB SVM-linear_{target}_{feature_comb} submit_dag_SVM-linear_SHIP_Stroop_NAI.submit\n")
            f.write(
                f"VARS SVM-linear_{target}_{feature_comb} feature_comb=\"{feature_comb}\" target=\"{target}\" req_cpu=\"8\"\n")

    for target in targets:
        for feature_comb in feature_combs:
            # Define SHAP job
            f.write(f"JOB SHAP_{target}_{feature_comb} submit_dag_SHAP_SVM-linear_SHIP_Stroop_NAI.submit\n")
            f.write(f"VARS SHAP_{target}_{feature_comb} feature_comb=\"{feature_comb}\" target=\"{target}\"\n")

    for target in targets:
        for feature_comb in feature_combs:
            # Define the parent-child relationship
            f.write(f"PARENT SVM-linear_{target}_{feature_comb} CHILD SHAP_{target}_{feature_comb}\n")
