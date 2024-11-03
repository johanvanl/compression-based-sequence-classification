from enum import Enum, auto
from collections import Counter

import numpy as np

from compression_classifier.classifier.helpers.np_counter import NPCounter
from compression_classifier.classifier.helpers.system_id_bidict import StaticNoCopySystemIdBiDict
from compression_classifier.classifier.conc_dict.utils import byte_hash

class EScoring(Enum):

    TF = auto()
    TF_SUBLINEAR = auto()
    TF_ICF = auto()
    TF_ICF_SUBLINEAR = auto()
    PPMI = auto()
    PPMI_SUBLINEAR = auto()

def get_class_scores(e_scoring : EScoring, cls_samples : list[list[bytes]],
                     k : int) -> list[NPCounter]:
    '''
    Factory method to get class scores.

    Args:
        e_scoring (EScoring): The chosen scoring.
        cls_samples (list[list[bytes]]): A list, of lists per class, of samples.
                                         Obtain using 'split_into_classes'.
        k (int): k-mer size.

    Returns:
        list[NPCounter]: A list, per class, of NPCounter objects.
    '''
    match e_scoring:
        case EScoring.TF:
            return build_tf_cnts(cls_samples, k, False)
        case EScoring.TF_SUBLINEAR:
            return build_tf_cnts(cls_samples, k, True)
        case EScoring.TF_ICF:
            return build_tf_icf_cnts(cls_samples, k, False)
        case EScoring.TF_ICF_SUBLINEAR:
            return build_tf_icf_cnts(cls_samples, k, True)
        case EScoring.PPMI:
            return build_ppmi_cnts(cls_samples, k, alpha=1)
        case EScoring.PPMI_SUBLINEAR:
            return build_ppmi_cnts(cls_samples, k, alpha=0.75)
    raise ValueError('Unknown Scoring!')

def build_raw_cnts(cls_samples : list[list[bytes]], k : int, min_df : int
                   ) -> tuple[StaticNoCopySystemIdBiDict, list[Counter]]:
    num_classes = len(cls_samples)
    df = Counter()
    cnts = [Counter() for _ in range(num_classes)]
    for class_idx in range(num_classes):
        for C in cls_samples[class_idx]:
            doc_kmers = set()
            for i in range(len(C) - k + 1):
                kmer = byte_hash(C[i:i+k])
                doc_kmers.add(kmer)
                cnts[class_idx][kmer] += 1
            df.update(doc_kmers)
    return StaticNoCopySystemIdBiDict(df, min_cnt=min_df), cnts

def build_tf_cnts(cls_samples : list[list[bytes]], k : int, sublinear_tf : bool = False,
                  min_df : int = 2) -> list[NPCounter]:
    kmers_bidi, raw_cnts = build_raw_cnts(cls_samples, k, min_df=min_df)

    num_classes = len(cls_samples)

    cnts = []
    for class_idx in range(num_classes):
        cnt = NPCounter(kmers_bidi)
        cnt.add_counter(raw_cnts[class_idx])
        if sublinear_tf:
            # Avoids the division by zero error
            np.clip(cnt.array, a_min=1e-8, a_max=None, out=cnt.array)
            # Clip to non-negative after sublinear transform
            cnt.array = np.clip(np.log10(cnt.array) + 1, a_min=0, a_max=None)
        cnts.append(cnt)
    
    return cnts

def build_tf_icf_cnts(cls_samples : list[list[bytes]], k : int, sublinear_tf : bool = False,
                      min_df : int = 2) -> list[NPCounter]:
    kmers_bidi, raw_cnts = build_raw_cnts(cls_samples, k, min_df=min_df)
    
    num_classes = len(cls_samples)

    cnts = []
    for class_idx in range(num_classes):
        cnt = NPCounter(kmers_bidi)
        cnt.add_counter(raw_cnts[class_idx])
        if sublinear_tf:
            # Avoids the division by zero error
            np.clip(cnt.array, a_min=1e-8, a_max=None, out=cnt.array)
            # Clip to non-negative after sublinear transform
            cnt.array = np.clip(np.log10(cnt.array) + 1, a_min=0, a_max=None)
        cnts.append(cnt)

    # IDF
    idf = []
    for class_idx in range(num_classes):
        idf.append(cnts[class_idx].array)
    idf = np.array(idf)
    # Document Frequency
    idf = np.sum(np.clip(idf, a_min=0, a_max=1), axis=0)

    # Final transform
    idf = np.log10(num_classes / idf)    

    for class_idx in range(num_classes):
        np.multiply(cnts[class_idx].array, idf, out=cnts[class_idx].array)
    
    return cnts

def build_ppmi_cnts(cls_samples : list[list[bytes]], k : int, alpha : float = 0.75,
                   min_df : int = 2) -> list[NPCounter]:
    kmers_bidi, raw_cnts = build_raw_cnts(cls_samples, k, min_df=min_df)
    
    num_classes = len(cls_samples)

    cnts = []
    for class_idx in range(num_classes):
        cnt = NPCounter(kmers_bidi)
        cnt.add_counter(raw_cnts[class_idx])
        cnts.append(cnt)

    # PMI
    pmi = []
    for class_idx in range(num_classes):
        pmi.append(cnts[class_idx].array)
    pmi = np.array(pmi)

    # Smoothing
    np.add(pmi, 0.1, out=pmi)

    kmer_mar = np.sum(pmi, axis=0)
    class_mar = np.sum(pmi, axis=1)

    # Use alpha to raise probability of rare contexts,
    # thus lowering their PPMI
    np.power(kmer_mar, alpha, out=kmer_mar)

    total = np.log2(np.sum(class_mar))
    np.log2(kmer_mar, out=kmer_mar)
    np.log2(class_mar, out=class_mar)

    for class_idx in range(num_classes):
        den = ( kmer_mar - total ) + ( class_mar[class_idx] - total )
        pmi[class_idx] = np.clip(np.log2(pmi[class_idx]) - total - den, a_min=0, a_max=None)

    for class_idx in range(num_classes):
        cnts[class_idx].array = pmi[class_idx]
    
    return cnts
