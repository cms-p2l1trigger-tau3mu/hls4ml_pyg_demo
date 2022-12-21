# -*- coding: utf-8 -*-

"""
Created on 2021/4/24

@author: Siqi Miao
"""

from nbformat import write
import yaml
import shutil
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from itertools import product
from matplotlib import figure
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import check_matplotlib_support

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard._convert_np import make_np
from tensorboard.plugins.pr_curve.plugin_data_pb2 import PrCurvePluginData
from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.summary_pb2 import Summary


def log_epoch(epoch, phase, loss_dict, clf_probs, clf_labels, batch, writer=None, exp_probs=None, exp_labels=None):
    desc = f'[Epoch: {epoch}]: {phase}........., ' if batch else f'[Epoch: {epoch}]: {phase} finished, '
    for k, v in loss_dict.items():
        if not batch and writer is not None:
            writer.add_scalar(f'{phase}/{k}', v, epoch)
        desc += f'{k}: {v:.3f}, '
    if batch:
        return desc

    # # just quick debugging
    # max_val = torch.max(clf_probs)
    # min_val = torch.min(clf_probs)
    # print(f"clf_probs max: {max_val}, min: {min_val}")

    auroc = metrics.roc_auc_score(clf_labels, clf_probs)
    partial_auroc = metrics.roc_auc_score(clf_labels, clf_probs, max_fpr=0.001)
    fpr, recall, thres = metrics.roc_curve(clf_labels, clf_probs)
    # indices = get_idx_for_interested_fpr(fpr, [0.001, 0.001/10])
    # print(f"fpr: {fpr}")
    # print(f"recall: {recall}")
    # print(f"thres: {thres}")
    norm_factor = 31e6 # 31 MHz
    interested_rates = [1e3, 10e3, 30e3] # 1kHz, 10 kHz, 30 kHz
    interested_fprs = [rate/norm_factor for rate in interested_rates]
    indices = get_idx_for_interested_fpr(fpr, interested_fprs)
    interested_recalls = get_recall_for_interested_fpr(fpr, recall, interested_fprs)


    if writer is not None:
        writer.add_scalar(f'{phase}/AUROC/', auroc, epoch)
        writer.add_scalar(f'{phase}/Partial_AUROC/', partial_auroc, epoch)
        writer.add_scalar(f'{phase}/recall_on_max_fpr/', recall[indices[0]], epoch)
        writer.add_scalar(f'{phase}/recall_on_max_fpr_over_10/', recall[indices[1]], epoch)
        writer.add_roc_curve(f'ROC_Curve/{phase}', clf_labels, clf_probs, epoch)
        writer.add_roc_curve(f'TriggerRate_Curve/{phase}', clf_labels, clf_probs, epoch, to_trigger_rate=True)

    fig = PlotROC(fpr=fpr*31000, tpr=recall, roc_auc=auroc).plot().figure_  # kHz
    if writer is not None: writer.add_figure(f'TriggerRate/{phase}', fig, epoch)

    cm = metrics.confusion_matrix(clf_labels, y_pred=clf_probs > thres[indices[0]], normalize=None)
    fig = PlotCM(confusion_matrix=cm, display_labels=['Neg', 'Pos']).plot(cmap=plt.cm.Blues).figure_
    if writer is not None: writer.add_figure(f'Confusion Matrix - max_fpr/{phase}', fig, epoch)

    cm = metrics.confusion_matrix(clf_labels, y_pred=clf_probs > thres[indices[1]], normalize=None)
    fig = PlotCM(confusion_matrix=cm, display_labels=['Neg', 'Pos']).plot(cmap=plt.cm.Blues).figure_
    if writer is not None: writer.add_figure(f'Confusion Matrix - max_fpr_over_10/{phase}', fig, epoch)

    desc += f'auroc: {auroc:.3f}, '
    # desc += f'recall@1kHz: {recall[indices[0]]:.4f}, recall@10kHz: {recall[indices[1]]:.4f}, recall@30kHz: {recall[indices[2]]:.4f}'
    desc += f'recall@1kHz: {interested_recalls[0]:.4f}, recall@10kHz: {interested_recalls[1]:.4f}, recall@30kHz: {interested_recalls[2]:.4f}'


    if exp_probs is not None and exp_labels is not None and -1 not in exp_labels and -1 not in exp_probs:
        exp_auroc = metrics.roc_auc_score(exp_labels, exp_probs)
        desc += f', exp_auroc: {exp_auroc:.3f}'

        bkg_att_weights = exp_probs[exp_labels == 0]
        signal_att_weights = exp_probs[exp_labels == 1]
        desc += f', avg_bkg: {bkg_att_weights.mean():.3f}, avg_signal: {signal_att_weights.mean():.3f}'

        if writer is not None:
            writer.add_scalar(f'{phase}/Exp_AUROC/', exp_auroc, epoch)
            writer.add_histogram(f'{phase}/bkg_att_weights', bkg_att_weights, epoch)
            writer.add_histogram(f'{phase}/signal_att_weights', signal_att_weights, epoch)
            writer.add_scalar(f'{phase}/avg_bkg_att_weights/', bkg_att_weights.mean(), epoch)
            writer.add_scalar(f'{phase}/avg_signal_att_weights/', signal_att_weights.mean(), epoch)

    return desc, auroc, recall[indices[0]].item(), loss_dict['total']


def get_idx_for_interested_fpr(fpr, interested_fpr):
    res = []
    for each in interested_fpr:
        for i in range(1, fpr.shape[0]):
            if fpr[i] > each:
                res.append(i-1)
                break
    assert len(res) == len(interested_fpr)
    return res

def get_recall_for_interested_fpr(fpr_l, recall_l, interested_fprs: list):
    """
    We take into account that sklearn roc_curve skips threshold that
    aren't interesting, and we use linear projection to estimate proper
    recall value 
    """
    interested_recalls = []
    for interested_fpr_val in interested_fprs:
        for i in range(1, fpr_l.shape[0]):
            if fpr_l[i] > interested_fpr_val:
                x_start = fpr_l[i-1]
                x_end = fpr_l[i]
                y_start = recall_l[i-1]
                y_end = recall_l[i]
                grad = (y_end-y_start) / (x_end-x_start)
                x = interested_fpr_val - x_start
                interested_recall_val = grad*x + y_start
                interested_recalls.append(interested_recall_val)
                break
    assert len(interested_recalls) == len(interested_fprs)
    return interested_recalls

class Writer(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):

        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            self.add_scalar(k, v)

    def add_roc_curve(self, tag, labels, predictions, global_step=None,
                      num_thresholds=127, weights=None, walltime=None, to_trigger_rate=False):

        torch._C._log_api_usage_once("tensorboard.logging.add_pr_curve")
        labels, predictions = make_np(labels), make_np(predictions)
        self._get_file_writer().add_summary(
            Writer.roc_curve(tag, labels, predictions, num_thresholds, weights, to_trigger_rate),
            global_step, walltime)

    @staticmethod
    def roc_curve(tag, labels, predictions, num_thresholds=127, weights=None, to_trigger_rate=False):
        # weird, value > 127 breaks protobuf
        num_thresholds = min(num_thresholds, 127)
        data = Writer.compute_roc_curve(labels, predictions, num_thresholds, weights, to_trigger_rate)
        pr_curve_plugin_data = PrCurvePluginData(
            version=0, num_thresholds=num_thresholds).SerializeToString()
        plugin_data = SummaryMetadata.PluginData(
            plugin_name='pr_curves', content=pr_curve_plugin_data)
        smd = SummaryMetadata(plugin_data=plugin_data)
        tensor = TensorProto(dtype='DT_FLOAT',
                             float_val=data.reshape(-1).tolist(),
                             tensor_shape=TensorShapeProto(
                                 dim=[TensorShapeProto.Dim(size=data.shape[0]),
                                      TensorShapeProto.Dim(size=data.shape[1])]))
        return Summary(value=[Summary.Value(tag=tag, metadata=smd, tensor=tensor)])

    @staticmethod
    def compute_roc_curve(labels, predictions, num_thresholds=None, weights=None, to_trigger_rate=False):
        _MINIMUM_COUNT = 1e-7

        if weights is None:
            weights = 1.0

        # Compute bins of true positives and false positives.
        bucket_indices = np.int32(np.floor(predictions * (num_thresholds - 1)))
        float_labels = labels.astype(np.float)
        histogram_range = (0, num_thresholds - 1)
        tp_buckets, _ = np.histogram(
            bucket_indices,
            bins=num_thresholds,
            range=histogram_range,
            weights=float_labels * weights)
        fp_buckets, _ = np.histogram(
            bucket_indices,
            bins=num_thresholds,
            range=histogram_range,
            weights=(1.0 - float_labels) * weights)

        # Obtain the reverse cumulative sum.
        tp = np.cumsum(tp_buckets[::-1])[::-1]
        fp = np.cumsum(fp_buckets[::-1])[::-1]
        tn = fp[0] - fp
        fn = tp[0] - tp
        precision = tp / np.maximum(_MINIMUM_COUNT, tp + fp)
        recall = tp / np.maximum(_MINIMUM_COUNT, tp + fn)
        fpr = fp / np.maximum(_MINIMUM_COUNT, fp + tn)

        if to_trigger_rate:
            fpr = fpr * 310  # unit: 100 kHz

        return np.stack((tp, fp, tn, fn, recall, fpr))


class PlotROC(metrics.RocCurveDisplay):
    def plot(self, ax=None, *, name=None, **kwargs):
        check_matplotlib_support('RocCurveDisplay.plot')

        if ax is None:
            fig = figure.Figure()
            ax = fig.subplots()

        name = self.estimator_name if name is None else name

        line_kwargs = {}
        if self.roc_auc is not None and name is not None:
            line_kwargs["label"] = f"{name} (AUC = {self.roc_auc:0.3f})"
        elif self.roc_auc is not None:
            line_kwargs["label"] = f"AUC = {self.roc_auc:0.3f}"
        elif name is not None:
            line_kwargs["label"] = name

        line_kwargs.update(**kwargs)

        self.line_ = ax.plot(self.fpr, self.tpr, **line_kwargs)[0]
        ax.set_xlabel("Trigger Rate (kHz)")
        ax.set_ylabel("Signal Efficiency")
        ax.set_xlim(0, 100)  # 0 - 100 kHz

        if "label" in line_kwargs:
            ax.legend(loc='lower right')

        self.ax_ = ax
        self.figure_ = ax.figure
        return self


class PlotCM(metrics.ConfusionMatrixDisplay):
    def plot(self, *, include_values=True, cmap='viridis',
             xticks_rotation='horizontal', values_format=None,
             ax=None, colorbar=True):
        check_matplotlib_support("ConfusionMatrixDisplay.plot")

        if ax is None:
            fig = figure.Figure()
            ax = fig.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        self.text_ = None
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0

            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min

                if values_format is None:
                    text_cm = format(cm[i, j], '.2g')
                    if cm.dtype.kind != 'f':
                        text_d = format(cm[i, j], 'd')
                        if len(text_d) < len(text_cm):
                            text_cm = text_d
                else:
                    text_cm = format(cm[i, j], values_format)

                self.text_[i, j] = ax.text(
                    j, i, text_cm,
                    ha="center", va="center",
                    color=color)

        if self.display_labels is None:
            display_labels = np.arange(n_classes)
        else:
            display_labels = self.display_labels
        if colorbar:
            fig.colorbar(self.im_, ax=ax)
        ax.set(xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=display_labels,
               yticklabels=display_labels,
               ylabel="True label",
               xlabel="Predicted label")

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self
