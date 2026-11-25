"""Within-subject split: train, validate and test on the SAME person.

    from eeg_data.within_subject import within_subject
    d = within_subject('sub-03')
"""
from .common import state, stats_of, subject_data


def within_subject(s, avg_reps=True, include_test=False):
    """Data of subject s, normalised with the subject's own statistics.

    Returns a dict:
      X_train (n_train, 17, 60), idx_train   training concepts of subject s
      X_val   (n_val, 17, 60),   idx_val     validation concepts of subject s
      X_test  (200, 17, 60)                  only with include_test=True (final evaluation only)
    idx_* are row numbers of the 16,540 training images (order of eeg_data.common.train_files()).
    avg_reps=False keeps the repetitions: (n, 4, 17, 60) for train/val and (200, 80, 17, 60) for test.
    """
    parts = ('train', 'val') + (('test',) if include_test else ())
    return subject_data(s, stats_of(state().cfg['own'][s]), parts, avg_reps)
