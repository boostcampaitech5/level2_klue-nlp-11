from torchmetrics.classification import MulticlassF1Score, MulticlassAveragePrecision, MulticlassAccuracy
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# klue_re_micro_f1 = MulticlassF1Score(num_classes=30, average='micro', ignore_index=0)
klue_re_auprc = MulticlassAveragePrecision(num_classes=30)
klue_re_acc = MulticlassAccuracy(num_classes=30, average='macro')

if torch.cuda.is_available():
    # klue_re_micro_f1.cuda()
    klue_re_auprc.cuda()
    klue_re_acc.cuda()


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        'no_relation', 'org:top_members/employees', 'org:members', 'org:product', 'per:title', 'org:alternate_names',
        'per:employee_of', 'org:place_of_headquarters', 'per:product', 'org:number_of_employees/members',
        'per:children', 'per:place_of_residence', 'per:alternate_names', 'per:other_family', 'per:colleagues',
        'per:origin', 'per:siblings', 'per:spouse', 'org:founded', 'org:political/religious_affiliation',
        'org:member_of', 'per:parents', 'org:dissolved', 'per:schools_attended', 'per:date_of_death',
        'per:date_of_birth', 'per:place_of_birth', 'per:place_of_death', 'org:founded_by', 'per:religion'
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return f1_score(labels.cpu(), preds.cpu(), average="micro", labels=label_indices)


# def klue_re_auprc(probs, labels):
#     """KLUE-RE AUPRC (with no_relation)"""
#     probs = probs.cpu().numpy()
#     labels = labels.cpu().numpy()
#     scores = []
#     for i in range(30):
#         precision, recall, _ = sklearn.metrics.precision_recall_curve(labels == i, probs[:, i])
#         scores.append(sklearn.metrics.auc(recall, precision))
#     return np.mean(scores) * 100.0