# predict_with_bilstm.py
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

def set_gpu_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

def batched_predict(model, X, batch_size=16):
    ds = tf.data.Dataset.from_tensor_slices(X.astype(np.float32))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    y = model.predict(ds, verbose=1).flatten()
    return y

def build_lr_features(ages, y_bs_pred):
    ages = np.asarray(ages, dtype=np.float32)
    y_bs_pred = np.asarray(y_bs_pred, dtype=np.float32)
    feats = np.column_stack([
        ages,
        y_bs_pred,
        ages ** 2,
        y_bs_pred ** 2,
        ages * y_bs_pred,
        np.log(ages + 1.0),
        1.0 / (y_bs_pred + 1e-6),
    ])
    return feats

def main():
    parser = argparse.ArgumentParser(description="Predict embryo score/quality using trained BiLSTM best model.")
    parser.add_argument("--features-npy", type=str, required=True, help="Path to features npy (shape: [N, 33, 11*feature_dim]).")
    parser.add_argument("--model-path", type=str, default="bilstm_quality_model.h5", help="Path to trained BiLSTM model (.h5).")
    parser.add_argument("--output-csv", type=str, default="bilstm_predictions.csv", help="Output CSV path.")
    parser.add_argument("--ages-npy", type=str, default=None, help="Optional ages npy to enable quality classification.")
    parser.add_argument("--scaler-path", type=str, default=None, help="Optional sklearn scaler (joblib) for logistic regression.")
    parser.add_argument("--logreg-path", type=str, default=None, help="Optional sklearn logistic regression (joblib) for quality classification.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for prediction.")
    args = parser.parse_args()

    set_gpu_memory_growth()

    # Load features
    X = np.load(args.features_npy)
    if X.ndim != 3 or X.shape[1] != 33:
        raise ValueError(f"Expected features shape [N, 33, M], got {X.shape}")

    # Load model
    model = tf.keras.models.load_model(args.model_path)

    # Predict BS scores (regression)
    y_bs_pred = batched_predict(model, X, batch_size=args.batch_size)

    data = {
        "index": np.arange(len(y_bs_pred), dtype=int),
        "bs_pred": y_bs_pred
    }

    # Optional: classify quality via external scaler + logistic regression
    prob_good = None
    pred_label = None
    if args.ages_npy and args.scaler_path and args.logreg_path:
        import joblib
        ages = np.load(args.ages_npy).astype(np.float32)
        if ages.shape[0] != y_bs_pred.shape[0]:
            raise ValueError("ages-npy length must match number of samples in features-npy.")
        feats = build_lr_features(ages, y_bs_pred)
        scaler = joblib.load(args.scaler_path)
        clf = joblib.load(args.logreg_path)
        feats_scaled = scaler.transform(feats)
        proba = clf.predict_proba(feats_scaled)[:, 1]  # prob Good
        prob_good = proba
        pred_label = (proba >= 0.5).astype(int)
        data["age"] = ages
        data["prob_good"] = prob_good
        data["prob_poor"] = 1.0 - prob_good
        data["pred_label"] = pred_label  # 1 = Good, 0 = Poor
        print("Quality classification enabled via external logistic regression.")
    else:
        print("No logistic regression provided. Output contains only BS regression (bs_pred).")

    # Save CSV
    df = pd.DataFrame(data)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")

if __name__ == "__main__":
    main()