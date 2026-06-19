import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, roc_auc_score
from rule_engine import StocksRuleEngine
from constants import MODEL_PATH, TODAY, MODEL_SCHEMA_METADATA, FEATURE_COLUMNS

def generate_training_labels(
    df: pd.DataFrame, 
    horizon: int = 20, 
    profit_threshold: float = 0.08
) -> pd.DataFrame:
    """
    Generates binary targets using a highly optimized, fully vectorized forward window.
    Replaces the O(N * horizon) nested loops with an O(N) double-shift rolling max.
    """
    processed_groups = []
    
    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values(by="Date").copy()
        
        # Vectorized Forward-Looking Rolling Max
        future_max = (
            group["Close"]
            .shift(-1)
            .rolling(window=horizon, min_periods=horizon)
            .max()
            .shift(-(horizon - 1))
        )
        
        # Binary target vector assignment via fast vectorized check
        group["Target_Label"] = (
            (future_max - group["Close"]) / group["Close"] >= profit_threshold
        ).astype(int)
        
        # Drop the last 'horizon' rows as their forward window is cut off by the dataset boundary
        group = group.iloc[:-horizon]
        if not group.empty:
            processed_groups.append(group)
            
    return pd.concat(processed_groups, ignore_index=True) if processed_groups else pd.DataFrame()

def run_offline_model_training(raw_universe_df: pd.DataFrame, pipeline_instance: StocksRuleEngine):
    """
    Executes a high-speed, expanding window walk-forward validation matrix.
    Tracks pure alpha edge metrics and regularizes final model via median tree convergence.
    """
    print("\n" + "="*80)
    print("🚀 INITIALIZING RESEARCH-MODE EXPANDING WINDOW VALIDATION SUITE")
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
    
    # 2. High-Speed Vectorized Target Profiling
    print("[TRAIN] Extracting target velocity matrix via vectorized rolling window...")
    labeled_df = generate_training_labels(gold_df, horizon=20, profit_threshold=0.08)
    
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
    
    print(f"[WALK-FORWARD] Spawning 5-Fold Calendar-Block Analytics Matrix...\n")
    
    for fold in range(5):
        # Calculate strict deterministic chronological boundaries
        train_end = fold_size * (fold + 1)
        val_end = fold_size * (fold + 2)
        
        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[train_end:val_end]
        
        # High-speed vectorized filtering via pandas .isin()
        train_df = training_pool[training_pool["Date"].isin(train_dates)]
        val_df = training_pool[training_pool["Date"].isin(val_dates)]
        
        X_tr, y_tr = train_df[active_features], train_df["Target_Label"]
        X_va, y_va = val_df[active_features], val_df["Target_Label"]
        
        # Safe structural boundary guard
        if len(y_tr.unique()) < 2 or len(y_va.unique()) < 2:
            print(f" ⚠ [FOLD {fold+1}] Suspended due to structural outcome invariance.")
            continue
            
        # Baseline Reference Generation
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X_tr, y_tr)
        dummy_probs = dummy.predict_proba(X_va)[:, 1]
        baseline_logloss = log_loss(y_va, dummy_probs)
        
        # Instantiate Single-Threaded Histogram-Method Model
        fold_model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric=["logloss", "auc"], 
            early_stopping_rounds=35, # <-- MODIFICATION 3: Increased from 20 to allow Folds 1 & 2 to learn
            random_state=42,
            n_jobs=1,              
            tree_method="hist"     
        )
        
        fold_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        val_preds = fold_model.predict_proba(X_va)[:, 1]
        
        # Alpha Ranking Metric Computations
        fold_auc = roc_auc_score(y_va, val_preds)
        base_rate = float(y_va.mean())
        
        # Calculate Precision Concentration at top 10% (Live portfolio selector mock)
        eval_df = pd.DataFrame({"true": y_va, "pred": val_preds}).sort_values("pred", ascending=False)
        top_10_pct_count = max(1, int(len(eval_df) * 0.10))
        precision_top_10 = float(eval_df.head(top_10_pct_count)["true"].mean())
        
        # <-- MODIFICATION 2: Explicit Pure Alpha Edge Tracking
        alpha_edge = precision_top_10 - base_rate
        
        print(f" └─> [FOLD {fold+1}/5] Segment: {min(val_dates)} to {max(val_dates)}")
        print(f"     • Footprint: TrainDays={len(train_dates)} ValDays={len(val_dates)} | TrainRows={len(X_tr)} ValRows={len(X_va)}")
        print(f"     • Base Val Ratio: {base_rate:.4f} | Baseline Logloss: {baseline_logloss:.5f}")
        print(f"     • Out-of-Sample AUC: {fold_auc:.4f} | Precision@Top10%: {precision_top_10:.4f}")
        print(f"     • Pure Strategy Alpha Edge: {alpha_edge:+.4f}")
        print(f"     • Converged at Tree Iteration: {fold_model.best_iteration}\n")
        
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
            "alpha_edge": alpha_edge, # Saved directly to audit telemetry
            "best_iteration": int(fold_model.best_iteration)
        })

    # 5. Compile Master Model on Complete Production Footprint Using Median
    # <-- MODIFICATION 1: Switched from np.mean to np.median to prevent outlier distortion
    optimal_master_trees = int(np.median(best_iterations_tracker)) + 5 if best_iterations_tracker else 25
    print(f"[TRAIN] Multi-fold execution complete. Compiling final production network on global data map...")
    print(f"[TRAIN] Regularizing production capacity boundary via MEDIAN metric to exactly {optimal_master_trees} trees.")
    
    X_master = training_pool[active_features]
    y_master = training_pool["Target_Label"]
    
    master_model = XGBClassifier(
        n_estimators=optimal_master_trees, max_depth=3, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=1, tree_method="hist"
    )
    master_model.fit(X_master, y_master)
    
    # 6. Export Validated Model Binary and Metadata Audit Tracking Logs
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    master_model.save_model(MODEL_PATH)
    
    avg_auc = np.mean([r["auc"] for r in wf_records]) if wf_records else 0.0
    avg_p10 = np.mean([r["precision_top_10"] for r in wf_records]) if wf_records else 0.0
    avg_edge = np.mean([r["alpha_edge"] for r in wf_records]) if wf_records else 0.0
    
    metadata = {
        "compiled_on": TODAY,
        "features_schema": active_features,
        "global_samples_count": len(training_pool),
        "average_walk_forward_auc": float(avg_auc),
        "average_precision_top_10": float(avg_p10),
        "average_pure_alpha_edge": float(avg_edge), # Global tracking parameter
        "walk_forward_folds": wf_records,
        "feature_importance": dict(sorted(zip(active_features, [float(x) for x in master_model.feature_importances_]), key=lambda x: x[1], reverse=True))
    }
    
    with open(MODEL_SCHEMA_METADATA, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"="*80)
    print(f"🎯 SYSTEM REVIEW METRICS LOGGED SUCCESSFULLY")
    print(f"================================================================================")
    print(f"• Model Asset Path             : '{MODEL_PATH}'")
    print(f"• Average Walk-Forward AUC      : {avg_auc:.4f}")
    print(f"• Mean Expected Strategy Edge   : {avg_edge:+.4f} (Model Precision minus Base Rate)")
    print(f"================================================================================" + "\n")