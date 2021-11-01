import numpy as np
import pandas as pd
import os
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--folder-path", required=True, help="Output path.")
    parser.add_argument("-ord", "--ordinal", action='store_true', help="Flag to indicate if it's an ordinal.")
    return parser


def get_severity_score_classification(arr):
    return np.sum(arr * np.repeat(np.arange(1, arr.shape[-1] + 1)[None, ], arr.shape[0], axis=0), -1) - 1


def get_severity_score_ordinal(arr):
    return arr.sum(1)


def proba_to_label(probas):
    predict_levels = probas > 0.5
    predicted_labels = np.sum(predict_levels, axis=1)
    return predicted_labels


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    for pred_file in ['predictions.csv', 'predictions_validation.csv']:
        df = pd.read_csv(os.path.join(args.folder_path, pred_file))
        # Remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        pred = np.array([eval(p) for p in df['pred']])

        # Generate MC column results
        if 'mc_epistemic_0' in df:
            n_it = 50
            mc_pred = []
            for mc in range(n_it):
                col_name = "mc_epistemic_" + str(mc)
                mc_pred.append([eval(j) for j in df[col_name]])
            df['pred_mc'] = [list(p) for p in np.array(mc_pred).mean(0)]
            if args.ordinal:
                df['predicted_mc_class'] = proba_to_label(np.array(mc_pred).mean(0))
                df['soft_mc_prediction'] = get_severity_score_ordinal(np.array(mc_pred).mean(0))

            else:
                df['predicted_mc_class'] = np.array(mc_pred).mean(0).argmax(1)
                df['soft_mc_prediction'] = get_severity_score_classification(np.array(mc_pred).mean(0))

        if args.ordinal:
            print("Ordinal")
            df['predicted_class'] = proba_to_label(pred)
            df['soft_prediction'] = get_severity_score_ordinal(pred)
        else:
            print("Classification")
            df['predicted_class'] = pred.argmax(1)
            df['soft_prediction'] = get_severity_score_classification(pred)

        df.to_csv(os.path.join(args.folder_path, pred_file))
