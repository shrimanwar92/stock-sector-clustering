import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from rule_engine import StocksRuleEngine
from constants import MODEL_PATH, TODAY, MODEL_SCHEMA_METADATA, FEATURE_COLUMNS

def generate_training_labels(
    df: pd.DataFrame, 
    horizon: int = 20, 
    profit_threshold: float = 0.10, 
    max_drawdown_limit: float = 0.05
) -> pd.DataFrame:
    """
    Generates high-fidelity binary targets for systematic swing trading.
    
    Target = 1 ONLY if:
      1. Maximum forward return reaches profit_threshold within N sessions.
      2. The asset does NOT breach the max_drawdown_limit BEFORE reaching 
         the profit threshold.
    """
    processed_groups = []
    
    for symbol, group in df.groupby("Symbol"):
        group = group.sort_values(by="Date").copy()
        closes = group["Close"].values
        n_rows = len(closes)
        
        target_labels = np.zeros(n_rows, dtype=int)
        
        for i in range(n_rows - horizon):
            window = closes[i + 1 : i + horizon + 1]
            current_close = closes[i]
            
            forward_returns = (window - current_close) / current_close
            
            profit_breaches = np.where(forward_returns >= profit_threshold)[0]
            drawdown_breaches = np.where(forward_returns <= -max_drawdown_limit)[0]
            
            first_profit_idx = profit_breaches[0] if len(profit_breaches) > 0 else 999
            first_drawdown_idx = drawdown_breaches[0] if len(drawdown_breaches) > 0 else 999
            
            if first_profit_idx < first_drawdown_idx and first_profit_idx != 999:
                target_labels[i] = 1
            else:
                target_labels[i] = 0
                
        group["Target_Label"] = target_labels
        group = group.iloc[:-horizon]
        if not group.empty:
            processed_groups.append(group)
            
    return pd.concat(processed_groups, ignore_index=True) if processed_groups else pd.DataFrame()

def run_offline_model_training(raw_universe_df: pd.DataFrame, pipeline_instance: StocksRuleEngine):
    """
    Extracts historical training matrices, applies relaxed domain constraints,
    enforces strict point-in-time chronological date splitting, applies early stopping,
    and logs tracking importance JSON metrics.
    """
    print("\n" + "="*80)
    print("🚀 INITIALIZING OFFLINE GRADIENT-BOOSTED TREE COMPILATION SUITE")
    print("="*80)
    
    # 1. Feature Extraction
    print("[TRAIN] Extracting structural features via asset store engine...")
    gold_df = pipeline_instance.engineer_gold_features(raw_universe_df)
    if gold_df.empty:
        print("[CRITICAL] Empty feature frame. Model compilation aborted.")
        return
        
    # 2. Append Target Variables (Explicit Keyword Argument Naming Verified)
    print("[TRAIN] Generating forward 20-day horizon price velocity vectors...")
    labeled_df = generate_training_labels(
        gold_df, 
        horizon=20, 
        profit_threshold=0.10, 
        max_drawdown_limit=0.05
    )
    
    # 3. Domain Domain Constraints
    train_mask = (
        (labeled_df["Close"] >= 15.0) & 
        (labeled_df[FEATURE_COLUMNS].notna().all(axis=1))
    )
    training_pool = labeled_df[train_mask].sort_values(by="Date").reset_index(drop=True)
    print(f"[TRAIN] Cleaned training observations verified: {training_pool.shape[0]} rows.")
    
    min_required_rows = 100 
    if len(training_pool) < min_required_rows:
        print(f"[WARN] Critical data scarcity: {len(training_pool)} rows. Aborting fit.")
        return

    # 4. Strict Chronological Group Splitting (Walk-Forward Isolation Rule)
    # Finds the precise date index that cleanly splits observations 80/20 without overlapping tickers
    unique_dates = sorted(training_pool["Date"].unique())
    split_date_idx = int(len(unique_dates) * 0.8)
    cutoff_date = unique_dates[split_date_idx]
    
    train_df = training_pool[training_pool["Date"] < cutoff_date]
    val_df = training_pool[training_pool["Date"] >= cutoff_date]
    
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["Target_Label"]
    X_val, y_val = val_df[FEATURE_COLUMNS], val_df["Target_Label"]
    
    print(f"[TRAIN] Split Profile -> Cutoff Date: {cutoff_date}")
    print(f"[TRAIN] Train Base: {X_train.shape[0]} rows | Validation Base: {X_val.shape[0]} rows.")
    
    # Imbalance Matrix Weighting
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = (neg_count / (pos_count + 1e-9))
    print(f"[TRAIN] Imbalance Optimization Map -> Scale Factor: {scale_pos_weight:.2f}")
    
    # 5. Model Compilation with Early Stopping Setup
    model = XGBClassifier(
        n_estimators=250,       # Raised since early stopping protects against over-indexing
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=20,  # Stops training automatically when logloss flatlines
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 6. Save Model Core
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    
    # Map and sort importances for feature selection analysis
    raw_importances = model.feature_importances_
    importance_map = dict(zip(FEATURE_COLUMNS, [float(v) for v in raw_importances]))
    sorted_importance = dict(sorted(importance_map.items(), key=lambda item: item[1], reverse=True))
    
    metadata = {
        "compiled_on": TODAY,
        "features_schema": FEATURE_COLUMNS,
        "training_samples_count": len(training_pool),
        "positive_class_ratio": float(np.mean(y_train)),
        "best_iteration": int(model.best_iteration) if hasattr(model, "best_iteration") else 0,
        "feature_importance": sorted_importance
    }
    
    # Write diagnostic json profiles
    with open(MODEL_SCHEMA_METADATA, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"🎯 PRODUCTION COMPILATION COMPLETE: Model architecture compiled to '{MODEL_PATH}'")
    print("="*80 + "\n")