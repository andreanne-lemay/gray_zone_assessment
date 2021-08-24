import os
import copy
import pandas as pd
import torch
from tqdm import tqdm
from monai.transforms import Activations, Compose
from gray_zone.loader import Dataset


def evaluate_model(model: torch.nn.Module,
                   loader: Dataset,
                   output_path: str,
                   device: str,
                   act: Activations,
                   transforms: Compose = None,
                   df: pd.DataFrame = None,
                   is_mc: bool = True,
                   image_colname: str = 'image',
                   suffix: str = None) -> pd.DataFrame:
    """ Evaluate model on test set. """
    best_model_path = os.path.join(output_path, "best_metric_model.pth")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    images = []
    y_pred = []
    if is_mc:
        UNCERTAINTY_LST = ['epistemic']
        mc_preds = {unc_type: [] for unc_type in UNCERTAINTY_LST}

    with torch.no_grad():
        for batch_idx, test_data in enumerate(tqdm(loader)):
            test_images, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
            )

            pred_prob = act(model(test_images)).detach().cpu().numpy().tolist()
            test_labels = test_labels.detach().cpu().numpy()

            for i in range(len(pred_prob)):
                y_pred.append(pred_prob[i])
                images.append(test_data[2][i])

            # Store multiple predictions if is_mc is activated
            if is_mc:
                mc_pred = {}
                for unc_type in UNCERTAINTY_LST:
                    mc_pred[unc_type] = monte_carlo_it(model, test_images, 50, act)
                    for i in range(len(pred_prob)):
                        mc_preds[unc_type].append([p[i] for p in mc_pred[unc_type]])

    pred_dict = {"pred": y_pred, image_colname: images}
    # Store multiple predictions if is_mc is activated
    if is_mc:
        for unc_type in UNCERTAINTY_LST:
            for mc_it in range(len(mc_preds[unc_type][0])):
                pred_dict["mc_" + unc_type + "_" + str(mc_it)] = [p[mc_it] for p in mc_preds[unc_type]]
    pred_df = pd.DataFrame().from_dict(pred_dict)

    # Join predictions to test metadata for analysis and remove unnamed columns
    df = df.join(pred_df.set_index(image_colname), on=image_colname)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Save results in csv file
    predictions_path = os.path.join(output_path, "predictions" + suffix + ".csv")
    df.to_csv(predictions_path)

    return df


def monte_carlo_it(model: torch.nn.Module,
                   input_data: torch.Tensor,
                   n_it: int,
                   act: Activations) -> list:
    """ Activate dropout and generate n_it Monte Carlo iterations. """
    model = copy.deepcopy(model)
    input_data = copy.deepcopy(input_data)
    pred_lst = []

    with torch.no_grad():
        # Activate dropout during inference
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        for mc_it in range(n_it):
            pred = act(model(input_data))
            pred_lst.append(pred.detach().cpu().numpy().tolist())

    return pred_lst
