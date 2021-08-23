import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import click
from gray_zone.models.coral import proba_to_label
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, cohen_kappa_score, confusion_matrix, \
    classification_report, precision_score, recall_score
from scipy.stats import spearmanr, kendalltau


def combine_by_patient(df: pd.DataFrame):
    if 'mc_pred' not in df:
        df['mc_pred'] = [[0]] * len(df)

    df = df.groupby('patient').agg(
        score=pd.NamedAgg(column='score', aggfunc='mean'),
        image=pd.NamedAgg(column='image', aggfunc='max'),
        label=pd.NamedAgg(column='label', aggfunc='mean'),
        # original_labels=pd.NamedAgg(column='original_labels', aggfunc='mean'),
        patient_diff=pd.NamedAgg(column='score', aggfunc=lambda x: np.diff(x)),
        patient_mean=pd.NamedAgg(column='score', aggfunc='mean'),
        img_per_patient=pd.NamedAgg(column='label', aggfunc=lambda x: len(x)),
        mc_pred=pd.NamedAgg(column='mc_pred', aggfunc=lambda x: list(np.array(x).mean(0))),
        soft_nclass=pd.NamedAgg(column='soft_nclass', aggfunc=lambda x: list(np.array(x).mean(0))),
        pred=pd.NamedAgg(column='pred', aggfunc=lambda x: list(np.array(x).mean(0))))

    df = df.explode('patient_diff')
    df['patient_diff'] = df['patient_diff'].astype('float')

    return df


def get_mc_unc(df):
    mean_pred = df['mc_pred'].mean(0)
    df['pred_entr'] = -np.sum(mean_pred * np.log(mean_pred), axis=1)
    df['pred_var'] = np.mean(np.var(df['mc_pred'], axis=0), axis=-1)
    df['score_range'] = np.max(df['score'], axis=0) - np.min(df['score'], axis=0)
    df['score_var'] = np.var(df['score'], axis=0)


def load_pred(df, colname='pred'):
    return list(np.array([eval(str(j)) for j in df[colname]]))


def load_mc_pred(df, n_it=50):
    mc_pred = []
    for i in range(n_it):
        col_name = "mc_epistemic_" + str(i)
        mc_pred.append([eval(j) for j in df[col_name]])
    return list(np.array(mc_pred).swapaxes(0, 1))


def get_classification_score(arr):
    return np.sum(arr * np.repeat(np.arange(1, arr.shape[-1] + 1)[None, ], arr.shape[0], axis=0), -1) - 1


def get_regression_nclass_score(pred):
    norm_score = (pred - pred.min()) / (pred - pred.min()).max()
    multi_score = np.zeros((len(pred), 2))
    multi_score[:, 0] = 1 - norm_score
    multi_score[:, -1] = norm_score
    return multi_score


def get_bin_label(df, model_type, thr=None):
    pred = np.array(load_pred(df))

    if 'classification' in model_type:
        df['bin_pred'] = list(pred.argmax(1))
    elif 'ordinal' in model_type:
        df['bin_pred'] = list(proba_to_label(pred))
    elif 'regression' in model_type:
        df['bin_pred'] = binarize(pred, thr)


def get_continuous_prediction(df, model_type):
    if 'mc' in model_type:
        pred = np.array(load_mc_pred(df)).mean(1)
        df['pred'] = list(pred)
    else:
        pred = np.array(df['pred'].tolist())

    if 'classification' in model_type:
        df['score'] = get_classification_score(pred)
        df['soft_nclass'] = list(pred)
    elif 'ordinal' in model_type:
        df['score'] = pred.sum(1)
        df['soft_nclass'] = list(get_regression_nclass_score(pred.sum(1)))
    elif 'regression' in model_type:
        df['score'] = pred.flatten()
        df['soft_nclass'] = list(get_regression_nclass_score(pred.flatten()))
    else:
        raise ValueError("Invalid model type name. Choices: 'classification', 'ordinal', 'regression'.")


def dist_3class(df, ax, gt_dict, fontsize=20):
    label = 'label'
    df['gt_label_str'] = [gt_dict[str(i)] for i in df[label]]
    sns.boxplot(x='gt_label_str', y='score', data=df, ax=ax, order=gt_dict.values())
    sns.swarmplot(y='score', x="gt_label_str", data=df,
                  color="black", edgecolor="gray", ax=ax,  order=gt_dict.values())
    spearman = spearmanr(df[label], df['score'])[0]
    ax.set_xlabel('', fontsize=fontsize)
    ax.set_ylabel('', fontsize=fontsize)
    ax.set_ylabel('Predicted severity score', fontsize=fontsize)
    ax.set_title(f"Prediction distribution (n={len(df)}), \n" + "\u03C1 = {:.3f}".format(spearman),
                 fontsize=fontsize + 2)


def patient_distance(df, ax, gt_dict, fontsize=20):
    df['gt_label_str'] = [gt_dict[str(i)] for i in df['label']]
    df_2img = df[df['img_per_patient'] > 1]
    for idx, label in enumerate(gt_dict.values()):
        ax.plot(df_2img[df_2img['label'] == idx]['patient_mean'],
                df_2img[df_2img['label'] == idx]['patient_diff'], "o", label=label)
    ax.axhline(df_2img['patient_diff'].mean(), linestyle="--")
    ax.axhline(df_2img['patient_diff'].mean() - df_2img['patient_diff'].std() * 1.96, linestyle="--")
    ax.axhline(df_2img['patient_diff'].mean() + df_2img['patient_diff'].std() * 1.96, linestyle="--")
    ax.set_xlabel('Mean', fontsize=fontsize)
    ax.set_ylabel('Difference', fontsize=fontsize)
    ax.legend()
    ax.set_ylim(-np.round(df['score'].max()), min(2, np.round(df['score'].max())))
    ax.set_title(f"Bland-Altman (n={len(df_2img)}),\n Avg. abs. diff.={abs(df_2img['patient_diff']).mean():.2f}",
                 fontsize=fontsize + 2)


def roc(df, ax, n_class, fontsize=20):
    y_true_bin = label_binarize(df['label'], classes=np.arange(int(n_class)))
    fpr, tpr, thr, opt, roc_auc = dict(), dict(), dict(), dict(), dict()

    pred = np.array(load_pred(df, 'soft_nclass'))
    for i in [0, -1]:
        fpr[i], tpr[i], thr[i] = roc_curve(y_true_bin[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        optimal_idx = np.argmax(tpr[i] - fpr[i])
        opt[i] = thr[i][optimal_idx]
    lw = 2
    ax.plot(fpr[0], tpr[0],
            lw=lw, label='Normal (area = %0.2f)' % roc_auc[0])
    ax.plot(fpr[-1], tpr[-1],
            lw=lw, label='Pre-cancer/Cancer (area = %0.2f)' % roc_auc[-1])
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=fontsize)
    ax.set_title(f"AUROC curve (n={len(df)})", fontsize=fontsize + 2)
    ax.legend(loc="lower right", fontsize=fontsize - 2)
    return opt


def plot_graphs(df, n_class, output_path, model_type, name, label_dict=None):
    fig, ax = plt.subplots(1, 4)
    fig.set_size_inches(32, 6)
    if label_dict is not None:
        gt_dict = json.load(open(label_dict, 'r'))
    else:
        gt_dict = {str(i): i for i in range(df['label'].max() + 1)}

    patient_distance(df, ax[2], gt_dict)
    dist_3class(df, ax[0], gt_dict)
    thresholds = roc(df, ax[1], n_class)
    pred = np.array(load_pred(df, 'soft_nclass'))
    conv_thr = [df['score'][pred[:, i] == thresholds[i]].tolist()[0] for i in [0, list(thresholds.keys())[-1]]]
    confusion_mat(df, ax[3], conv_thr, model_type)

    plt.tight_layout()
    fig.savefig(os.path.join(output_path, name + '.png'))
    plt.show()


def binarize(soft_pred, thr):
    gt_max = soft_pred.max()
    gt_min = soft_pred.min()
    n_class = 3
    thr.insert(0, gt_min)
    thr.append(gt_max)
    c_pred = []
    for s in soft_pred:
        for t in range(n_class):
            if (s <= thr[t + 1]) and (s >= thr[t]):
                c_pred.append(t)
                break
    return np.array(c_pred)


def confusion_mat(df, ax, thr, model_type, fontsize=20):
    label = 'label'
    get_bin_label(df, model_type, thr)
    pred_label = df['bin_pred']
    cm = confusion_matrix(df[label], pred_label)
    cm_df = pd.DataFrame(cm, index=np.arange(1, cm.shape[0] + 1), columns=np.arange(1, cm.shape[0] + 1))
    sns.heatmap(cm_df, annot=True, ax=ax, fmt='.3g', annot_kws={"size": fontsize})
    ax.set_xlabel("Prediction", fontsize=fontsize)
    ax.set_ylabel("GT", fontsize=fontsize)
    ax.set_title("Confusion matrix", fontsize=fontsize)


@click.command()
@click.option('--output-path', '-o', required=True, help='Output path.')
@click.option('--predfile-path', '-p', required=True, help='Path to prediction file.')
@click.option('--model-type', '-t', default="classification", help='Path to prediction file.')
@click.option('--gt-dict-path', '-g', default=None, help='Path to prediction file.')
@click.option('--label-colname', '-lc', default='label', help='Column name in csv associated to the labels.')
@click.option('--image-colname', '-ic', default='image', help='Column name in csv associated to the image.')
@click.option('--patient-colname', '-pc', default='patient',
              help='Column name in csv associated to the patient id.')
@click.option('--name', '-name', default='visualize', help='Name of file')
def generate_plots(output_path: str,
                   predfile_path: str,
                   label_colname: str,
                   image_colname: str,
                   patient_colname: str,
                   gt_dict_path: str,
                   name: str,
                   merge_per_patient: bool = True,
                   model_type: str = 'classification'):

    df = pd.read_csv(predfile_path)
    df = df.rename(columns={label_colname: 'label', image_colname: 'image', patient_colname: 'patient'})
    # Change pred and mc pred format from string to array
    df['pred'] = load_pred(df)
    if 'mc' in model_type:
        df['mc_pred'] = load_mc_pred(df)

    get_continuous_prediction(df, model_type)
    df = combine_by_patient(df)

    plot_graphs(df, df.label.max() + 1, output_path=output_path, model_type=model_type, label_dict=gt_dict_path, name=name)


if __name__ == "__main__":
    generate_plots()
