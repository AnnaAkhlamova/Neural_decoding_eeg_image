"""Leave-one-subject-out (LOSO) split: train on other people, validate and test on people the model has never seen.

    from eeg_data.loso import loso
    L = loso('sub-03')
"""
import numpy as np

from .common import state, stats_of, subject_data


def folds():
    """The 10 LOSO folds: {test subject: {'val': [2 subjects], 'train': [7 subjects]}} (fixed, same for everyone)."""
    return {s: dict(val=f['val'], train=f['train']) for s, f in state().cfg['loso'].items()}


def loso(test_s, new_subject_norm='B', avg_reps=True, include_test=False):
    """LOSO fold for the new (test) subject test_s.

    Training subjects use their own statistics. The new subjects (2 validation subjects and the test subject) use
      'B' = pooled statistics of the 7 training subjects (default, nothing from the new subject is used), or
      'A' = their own statistics (assumes a short unlabelled recording of the new person is available).
    Returns a dict:
      X_train, idx_train, subj_train   7 training subjects, training concepts only
      X_val,   idx_val,   subj_val     2 validation subjects, validation concepts only
      X_test  (200, 17, 60)            the new subject; only with include_test=True (final evaluation only)
      folds                            the subject lists of this fold
    subj_* hold the subject name of every row; idx_* are row numbers of the 16,540 training images.
    """
    st = state()
    F = st.cfg['loso'][test_s]
    assert new_subject_norm in ('A', 'B'), "new_subject_norm must be 'A' or 'B'"

    def new(s):
        return stats_of(F['pooled']) if new_subject_norm == 'B' else stats_of(st.cfg['own'][s])

    Xt, It, St, Xv, Iv, Sv = [], [], [], [], [], []
    for t in F['train']:                                           # training subjects, training concepts only
        d = subject_data(t, stats_of(st.cfg['own'][t]), ('train',), avg_reps)
        Xt.append(d['X_train'])
        It.append(d['idx_train'])
        St += [t] * len(d['idx_train'])
    for v in F['val']:                                             # validation subjects, validation concepts only
        d = subject_data(v, new(v), ('val',), avg_reps)
        Xv.append(d['X_val'])
        Iv.append(d['idx_val'])
        Sv += [v] * len(d['idx_val'])
    out = dict(X_train=np.concatenate(Xt), idx_train=np.concatenate(It), subj_train=np.array(St),
               X_val=np.concatenate(Xv), idx_val=np.concatenate(Iv), subj_val=np.array(Sv),
               folds=dict(val=F['val'], train=F['train']))
    if include_test:
        out['X_test'] = subject_data(test_s, new(test_s), ('test',), avg_reps)['X_test']
    return out
