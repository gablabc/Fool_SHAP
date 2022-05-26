import argparse
import numpy as np
import os
from tqdm import tqdm

# Local imports
from utils import get_data, get_foreground_background, load_model
from utils import audit_detection


if __name__ == "__main__":

    # Parser initialization
    parser = argparse.ArgumentParser(description='Script for training models')
    parser.add_argument('--dataset', type=str, default='adult_income', help='Dataset: adult_income, compas, default_credit, marketing')
    parser.add_argument('--model', type=str, default='rf', help='Model: mlp, rf, gbt, xgb')
    args = parser.parse_args()

    np.random.seed(42)
    false_positive_rates = 0

    detections = 0.0
    for rseed in tqdm(range(5)):
        # Get the data
        filepath = os.path.join("datasets", "preprocessed", args.dataset)
        X_split, y_split, features, ordinal_encoder, ohe_encoder = \
                                            get_data(args.dataset, args.model, rseed=rseed)
        # Get B and F
        foreground, background = get_foreground_background(X_split, args.dataset)
        
        # Ordinally encode B and F
        background = ohe_encoder.transform(ordinal_encoder.transform(background))
        foreground = ohe_encoder.transform(ordinal_encoder.transform(foreground))

        # Load the model
        tmp_filename = f"{args.model}_{args.dataset}_{rseed}"
        model = load_model(args.model, "models", tmp_filename)

        # All background/foreground predictions
        b_pred = model.predict_proba(background)[:, [1]]
        f_pred = model.predict_proba(foreground)[:, [1]]

        for _ in range(1000):
            # Biased sampling
            biased_idx = np.random.choice(len(background), 200)
            subset_b_pred = b_pred[np.random.choice(len(background), 200)]
            subset_f_pred = f_pred[np.random.choice(len(foreground), 200)]
            
            false_positive_rates += audit_detection(b_pred, f_pred,
                                                    subset_b_pred, subset_f_pred, 0.01)
    false_positive_rates /= 5000
    print(false_positive_rates)
