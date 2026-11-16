import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


# Post-hoc scores: higher means more likely to be known / in-distribution
SCORES = {
    'msp': lambda logits: torch.softmax(logits, dim=1).max(dim=1)[0],
    'max_logit': lambda logits: logits.max(dim=1)[0],
    'energy': lambda logits: torch.logsumexp(logits, dim=1),
}


def auroc(known_scores, unknown_scores):
    labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    return roc_auc_score(labels, np.concatenate([known_scores, unknown_scores]))


def fpr_at_95_tpr(known_scores, unknown_scores):
    labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    fpr, tpr, _ = roc_curve(labels, np.concatenate([known_scores, unknown_scores]))
    return fpr[np.searchsorted(tpr, 0.95)]


def oscr(known_scores, unknown_scores, correct):
    """Open-Set Classification Rate: area under the correct classification rate vs. false positive rate curve."""
    scores = np.concatenate([known_scores, unknown_scores])
    order = np.argsort(-scores, kind='stable')
    is_correct = np.concatenate([correct, np.zeros(len(unknown_scores), dtype=bool)])[order]
    is_unknown = np.concatenate([np.zeros(len(known_scores), dtype=bool), np.ones(len(unknown_scores), dtype=bool)])[order]
    ccr = np.concatenate([[0.], np.cumsum(is_correct) / len(known_scores)])
    fpr = np.concatenate([[0.], np.cumsum(is_unknown) / len(unknown_scores)])
    return np.sum(np.diff(fpr) * (ccr[1:] + ccr[:-1]) / 2)
