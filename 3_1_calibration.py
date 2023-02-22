""" Compute the calibration of the detector, i.e. Table 1 in the paper """
import argparse
import numpy as np
import os
from tqdm import tqdm

# Local imports
from src.utils import get_data, get_foreground_background, load_model
from src.utils import audit_detection


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
        D_0, D_1 = get_foreground_background(X_split, args.dataset)
        
        # Ordinally encode B and F
        D_0 = ordinal_encoder.transform(D_0)
        D_1 = ordinal_encoder.transform(D_1)
    
        if ohe_encoder is not None:
            D_0 = ohe_encoder.transform(D_0)
            D_1 = ohe_encoder.transform(D_1)
        
        # Load the model
        tmp_filename = f"{args.model}_{args.dataset}_{rseed}"
        model = load_model(args.model, "models", tmp_filename)

        # All background/foreground predictions
        f_D_0 = model.predict_proba(D_0)[:, 1]
        f_D_1 = model.predict_proba(D_1)[:, 1]
        M = 200

        for _ in range(1000):
            f_S_0 = f_D_0[np.random.choice(len(f_D_0), M)]
            f_S_1 = f_D_1[np.random.choice(len(f_D_1), M)]
            
            false_positive_rates += audit_detection(f_D_0, f_D_1, f_S_0, f_S_1)
    false_positive_rates /= 5000
    print(100 * false_positive_rates)
