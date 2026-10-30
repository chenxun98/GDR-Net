"""Class-first, patient-second sampling for heterogeneous EEG cohorts."""
from collections import Counter
import torch
from torch.utils.data import WeightedRandomSampler


def class_key(sample, classification):
    if int(sample.det_label) == 0:
        return 'background'
    if classification:
        if int(sample.cls_label) < 0:
            raise ValueError('Class-balanced subtype sampling requires labeled positive windows.')
        return f'cls_{int(sample.cls_label)}'
    return 'seizure'


def class_patient_sampler(dataset, classification, num_samples=None, seizure_fraction=.5):
    if not 0 < float(seizure_fraction) < 1:
        raise ValueError('seizure_fraction must be strictly between 0 and 1.')
    keys=[(class_key(s,classification),str(s.patient_id)) for s in dataset.samples]
    if not keys:
        raise ValueError('Cannot sample an empty dataset.')
    windows=Counter(keys)
    patients=Counter(k[0] for k in windows)
    positive=[k for k in patients if k!='background']
    if not positive or 'background' not in patients:
        raise ValueError('Class/patient sampling requires both positive and background windows.')
    priors={k:float(seizure_fraction)/len(positive) for k in positive}
    priors['background']=1.0-float(seizure_fraction)
    weights=torch.tensor([priors[c]/(patients[c]*windows[(c,p)]) for c,p in keys],dtype=torch.float64)
    return WeightedRandomSampler(weights,int(num_samples or len(keys)),replacement=True)


