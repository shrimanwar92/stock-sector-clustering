import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, roc_auc_score
from rule_engine import StocksRuleEngine
from constants import MODEL_PATH, TODAY, MODEL_TRAINING_METADATA, FEATURE_COLUMNS, CALIBRATOR_MODEL
from sklearn.calibration import CalibratedClassifierCV
import joblib

def run_offline_model_training(raw_universe_df: pd.DataFrame, pipeline_instance: StocksRuleEngine):
    """
    Executes a high-speed, expanding window walk-forward validation matrix.
    Tracks pure alpha edge metrics and regularizes final model via median tree convergence.
    
    UPGRADE: Shifted completely to 3-Class Multi-Class Engine to eliminate shape alignment crashes.
    """
    print("\n" + "="*80)
    print("🚀 INITIALIZING RESEARCH-MODE EXPANDING WINDOW VALIDATION SUITE (3-CLASS)")
    print("="*80)
    
    # 1. Feature Engineering & Column Pruning
    print("[TRAIN] Engineering feature store arrays...")
    gold_df = pipeline_instance.engineer_gold_features(raw_universe_df)
    if gold_df.empty:
        print("[CRITICAL] Engineered data matrix is empty. Terminating execution.")
        return
        
    active_features = [
        f for f in FEATURE_COLUMNS 
        if f not in ["Feature_Delivery_Ratio", "Feature_Close_Strength"]
    ]
    
    # 2. UPGRADE: Extract Triple-Barrier Labels via Rule Engine
    print("[TRAIN] Extracting 3-Class path analysis matrix via López de Prado framework...")
    labeled_df = pipeline_instance.engineer_training_labels(gold_df, pt_horizon=20, pt_mult=2.5, sl_mult=1.5)
    
    # 3. Apply Operational Masks & Reset Index
    train_mask = (
        (labeled_df["Close"] >= 15.0) & 
        (labeled_df[active_features].notna().all(axis=1))
    )
    training_pool = labeled_df[train_mask].sort_values("Date").reset_index(drop=True)
    
    if len(training_pool) < 500:
        print(f"[WARN] Insufficient historical data pool ({len(training_pool)} rows). Aborting.")
        return

    print(f"[TRAIN] Active data footprint verified: {training_pool.shape[0]} rows across {len(training_pool['Date'].unique())} unique days.")

    # 4. Deterministic Expanding Window Slicing Framework
    unique_dates = sorted(training_pool["Date"].unique())
    n_dates = len(unique_dates)
    fold_size = n_dates // 6  # Slices timeline into 6 equal chronological blocks
    
    wf_records = []
    best_iterations_tracker = []
    probability_std_tracker = []
    
    print(f"[WALK-FORWARD] Spawning 5-Fold Calendar-Block Analytics Matrix...\n")
    
    for fold in range(5):
        train_end = fold_size * (fold + 1)
        val_end = fold_size * (fold + 2)
        
        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[train_end:val_end]
        
        train_df = training_pool[training_pool["Date"].isin(train_dates)]
        val_df = training_pool[training_pool["Date"].isin(val_dates)]
        
        # UPGRADE: Targets bound to Strategic_Label multi-class targets
        X_tr, y_tr = train_df[active_features], train_df["Strategic_Label"]
        X_va, y_va = val_df[active_features], val_df["Strategic_Label"]
        
        # Safe structural boundary guard (Must have instances of failure, stagnation, and success)
        if len(y_tr.unique()) < 3 or len(y_va.unique()) < 3:
            print(f" ⚠ [FOLD {fold+1}] Suspended due to structural multi-class outcome invariance.")
            continue
            
        # Baseline Reference Generation (Multi-class array logloss check)
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X_tr, y_tr)
        dummy_probs = dummy.predict_proba(X_va)
        baseline_logloss = log_loss(y_va, dummy_probs, labels=[0, 1, 2])
        
        # UPGRADE: Configured Multi-Class Objectives explicitly
        fold_model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss", 
            early_stopping_rounds=40,
            random_state=42,
            n_jobs=1,              
            tree_method="hist"     
        )
        
        fold_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        # Shape: (N, 3) -> [P(Failure), P(Stagnation), P(Success)]
        val_probs_all = fold_model.predict_proba(X_va)
        val_probs_success = val_probs_all[:, 2] 
        
        # UPGRADE: Multi-Class OVR Macro AUC tracking
        fold_auc = roc_auc_score(y_va, val_probs_all, multi_class="ovr", average="macro")
        base_rate = float((y_va == 2).mean()) # Target Success base rate
        
        # Calculate Precision Concentration at top 10% for Class 2 (Success predictions)
        eval_df = pd.DataFrame({"true": (y_va == 2).astype(int), "pred": val_probs_success}).sort_values("pred", ascending=False)
        top_10_pct_count = max(1, int(len(eval_df) * 0.10))
        precision_top_10 = float(eval_df.head(top_10_pct_count)["true"].mean())
        
        alpha_edge = precision_top_10 - base_rate
        
        print(f" └─> [FOLD {fold+1}/5] Segment: {min(val_dates)} to {max(val_dates)}")
        print(f"     • Footprint: TrainDays={len(train_dates)} ValDays={len(val_dates)} | TrainRows={len(X_tr)} ValRows={len(X_va)}")
        print(f"     • Target Success Ratio: {base_rate:.4f} | Baseline Logloss: {baseline_logloss:.5f}")
        print(f"     • Multi-Class Macro AUC: {fold_auc:.4f} | Precision@Top10% (Success): {precision_top_10:.4f}")
        print(f"     • Pure Strategy Alpha Edge: {alpha_edge:+.4f}")
        print(f"     • Converged at Tree Iteration: {fold_model.best_iteration}\n")
        print(f"     • Success Probability Std Dev = " f"{np.std(val_probs_all[:,2]):.4f}")

        success_std = np.std(val_probs_all[:,2])
        probability_std_tracker.append(success_std)
        
        best_iterations_tracker.append(fold_model.best_iteration)
        wf_records.append({
            "fold": fold + 1,
            "val_start": str(min(val_dates)),
            "val_end": str(max(val_dates)),
            "train_rows": int(len(X_tr)),
            "val_rows": int(len(X_va)),
            "positive_class_ratio_val": base_rate,
            "dummy_baseline_logloss": float(baseline_logloss),
            "auc": float(fold_auc),
            "precision_top_10": precision_top_10,
            "alpha_edge": alpha_edge,
            "best_iteration": int(fold_model.best_iteration)
        })

    # 5. Compile Master Model on Complete Production Footprint Using Median Tree Convergence Boundary
    good_iterations = [
        itr
        for itr, std in zip(best_iterations_tracker, probability_std_tracker)
        if std > 0.03
    ]

    optimal_master_trees = (
        int(np.mean(good_iterations))
        if good_iterations
        else 50
    )
    print(f"[TRAIN] Multi-fold execution complete. Compiling final production network on global data map...")
    print(f"[TRAIN] Regularizing production capacity boundary via MEDIAN metric to exactly {optimal_master_trees} trees.")
    
    X_master = training_pool[active_features]
    y_master = training_pool["Strategic_Label"]

    print(training_pool["Strategic_Label"].value_counts(normalize=True))
    
    # UPGRADE: Final Master Model converted to Multi-Class format
    master_model = XGBClassifier(
        n_estimators=optimal_master_trees, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_jobs=1, tree_method="hist"
    )
    master_model.fit(X_master, y_master)

    calibrator = CalibratedClassifierCV(
        estimator=master_model,
        method="isotonic",
        cv=3
    )

    calibrator.fit(X_master, y_master)
    
    # 6. Export Validated Model Binary and Metadata Audit Tracking Logs
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    master_model.save_model(MODEL_PATH)
    joblib.dump(calibrator, CALIBRATOR_MODEL)
    
    avg_auc = np.mean([r["auc"] for r in wf_records]) if wf_records else 0.0
    avg_p10 = np.mean([r["precision_top_10"] for r in wf_records]) if wf_records else 0.0
    avg_edge = np.mean([r["alpha_edge"] for r in wf_records]) if wf_records else 0.0
    
    metadata = {
        "compiled_on": TODAY,
        "features_schema": active_features,
        "global_samples_count": len(training_pool),
        "average_walk_forward_auc": float(avg_auc),
        "average_precision_top_10": float(avg_p10),
        "average_pure_alpha_edge": float(avg_edge),
        "walk_forward_folds": wf_records,
        "feature_importance": dict(sorted(zip(active_features, [float(x) for x in master_model.feature_importances_]), key=lambda x: x[1], reverse=True))
    }

    pipeline_instance.compile_and_save_health_contract(
        training_pool=training_pool,
        master_model=master_model,
        avg_auc=avg_auc,       
        avg_p10=avg_p10,       
        avg_edge=avg_edge,     
        wf_records=wf_records  
    )
    
    with open(MODEL_TRAINING_METADATA, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"="*80)
    print(f"🎯 SYSTEM REVIEW METRICS LOGGED SUCCESSFULLY")
    print(f"================================================================================")
    print(f"• Model Asset Path             : '{MODEL_PATH}'")
    print(f"• Average Walk-Forward AUC      : {avg_auc:.4f}")
    print(f"• Mean Expected Strategy Edge   : {avg_edge:+.4f} (Model Precision minus Base Rate)")
    print(f"================================================================================" + "\n")