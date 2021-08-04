"""A test function."""
import click
import json
import os
import pandas as pd
import torch

from gray_zone.loader import loader
from gray_zone.utils import load_transforms
from gray_zone.models.model import get_model
from gray_zone.train import train
from gray_zone.evaluate import evaluate_model
from gray_zone.loss import get_loss

def _run_model(output_path: str,
               param_path: str,
               data_path: str,
               csv_path: str,
               label_colname: str,
               image_colname: str,
               split_colname: str,
               patient_colname: str,
               model_type: str,
               transfer_learning: bool) -> None:
    """Function version of train"""
    # Create output directory if it doesn't exist
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # Create directory for checkpoints if it doesn't exist
    path_to_checkpoints = os.path.join(output_path, "checkpoints")
    if not os.path.isdir(path_to_checkpoints):
        os.makedirs(path_to_checkpoints)

    # Save configuration file in output directory
    param_dict = json.load(open(param_path, 'r'))
    param_dict['n_class'] = int(pd.read_csv(csv_path)[label_colname].max() + 1)
    json.dump(param_dict, open(os.path.join(output_path, "params.json"), 'w'), indent=4)

    # Record environment and CLI
    job_record = get_job_record(param_dict['seed'])
    save_job_record(output_dir, record=job_record, name='train_record.json')

    # Convert transforms from config file
    train_transforms = load_transforms(param_dict["train_transforms"])
    val_transforms = load_transforms(param_dict["val_transforms"])

    # Get train, val, test loaders and test dataframe
    train_loader, val_loader, test_loader, test_df = loader(data_path=data_path,
                                                            output_path=output_path,
                                                            train_transforms=train_transforms,
                                                            val_transforms=val_transforms,
                                                            metadata_path=csv_path,
                                                            label_colname=label_colname,
                                                            image_colname=image_colname,
                                                            split_colname=split_colname,
                                                            patient_colname=patient_colname,
                                                            train_frac=param_dict['train_frac'],
                                                            test_frac=param_dict['test_frac'],
                                                            seed=param_dict['seed'],
                                                            batch_size=param_dict['batch_size'],
                                                            balanced=param_dict['sampling'])

    # Get model
    model, act = get_model(architecture=param_dict['architecture'],
                           model_type=param_dict['model_type'],
                           dropout_rate=param_dict['dropout_rate'],
                           n_class=param_dict['n_class'],
                           device=param_dict['device'],
                           transfer_learning=transfer_learning,
                           output_dir=output_path)

    optimizer = torch.optim.Adam(model.parameters(), param_dict['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    loss_function = get_loss(param_dict['loss'], param_dict['n_class'])

    train(model=model,
          act=act,
          train_loader=train_loader,
          val_loader=val_loader,
          loss_function=loss_function,
          optimizer=optimizer,
          device=param_dict['device'],
          n_epochs=param_dict['n_epochs'],
          output_path=output_path,
          scheduler=scheduler,
          n_class=param_dict['n_class'],
          model_type=model_type,
          val_metric=param_dict['val_metric'])

    df = evaluate_model(model=model,
                        loader=test_loader,
                        output_path=output_path,
                        device=device,
                        act=act,
                        transforms=val_transforms,
                        df=test_df,
                        is_mc=param_dict['dropout_rate'] > 0)


@click.command()
@click.option('--output-path', '-o', required=True, help='Output path.')
@click.option('--param-path', '-p', required=True, help='Path to parameter file (.json).')
@click.option('--data-path', '-d', required=True, help='Path to data (directory where images are saved).')
@click.option('--csv-path', '-c', required=True, help='Path to csv file containing image name and labels.')
@click.option('--label-colname', '-lc', default='label', help='Column name in csv associated to the labels.')
@click.option('--image-colname', '-ic', default='image', help='Column name in csv associated to the image.')
@click.option('--split-colname', '-sc', default='dataset',
              help="Column name in csv associated to the train, val, test splits. Each image needs to be associated "
                   "with `val`,`train`, or `test`")
@click.option('--patient-colname', '-pc', default='patient',
              help='Column name in csv associated to the patient id.')
@click.option('--model-type', '-m', default='classification',
              help="Choice between 'classification', 'ordinal', 'regression'.")
@click.option('--transfer-learning', '-tf', default=False, help="Fine-tune training from other model.")
def run_model(output_path: str,
              param_path: str,
              data_path: str,
              csv_path: str,
              label_colname: str,
              image_colname: str,
              split_colname: str,
              patient_colname: str,
              model_type: str,
              transfer_learning: bool) -> None:
    """Train deep learning model using CLI. """
    _run_model(output_path, param_path, data_path, csv_path, label_colname, image_colname, split_colname,
               patient_colname, model_type, transfer_learning)
