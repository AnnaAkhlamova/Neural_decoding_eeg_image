"""Shared pieces for both split types: data access (Hugging Face or a local copy), config files, preprocessing.

Nothing here depends on the split type. The two split types live in their own modules:
    eeg_data/within_subject.py  ->  within_subject(...)
    eeg_data/loso.py            ->  loso(...)
"""
import gc
import hashlib
import json
import os
import pickle
import tempfile
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_settings = dict(
    repo_id=os.environ.get('HF_REPO_ID'),                         # 'org/dataset-name' on the Hugging Face Hub
    hf_token=os.environ.get('HF_TOKEN'),
    local_root=os.environ.get('EEG_LOCAL_ROOT') or os.environ.get('EDA_LOCAL_ROOT'),   # folder that mirrors the repo (eeg/17ch/..., images/...)
    work_dir=os.environ.get('EEG_WORK') or os.path.join(tempfile.gettempdir(), 'eeg_work'),
    config_dir=os.environ.get('EEG_CONFIG_DIR') or os.path.join(_REPO_ROOT, 'configs'),
)


def configure(repo_id=None, hf_token=None, local_root=None, work_dir=None, config_dir=None):
    """Set where the data comes from. Call once before using the loaders (or use environment variables).
    repo_id: Hugging Face dataset repo ('org/name'); hf_token: read token (on Colab the Secret HF_TOKEN is used automatically);
    local_root: use a local copy of the dataset instead of downloading; config_dir: folder with the two json files."""
    for k, v in dict(repo_id=repo_id, hf_token=hf_token, local_root=local_root, work_dir=work_dir, config_dir=config_dir).items():
        if v is not None:
            _settings[k] = v
    state.cache_clear()


# ------------------------------ data access ------------------------------
def _token():
    if _settings['hf_token']:
        return _settings['hf_token']
    try:
        from google.colab import userdata
        _settings['hf_token'] = userdata.get('HF_TOKEN')
    except Exception:
        pass
    if not _settings['hf_token']:
        raise RuntimeError('No Hugging Face token: add the Colab Secret HF_TOKEN, set the HF_TOKEN environment variable, '
                           'or call eeg_data.configure(hf_token=...)')
    return _settings['hf_token']


def get_file(rel):
    """rel = path inside the dataset, e.g. 'eeg/17ch/sub-01/preprocessed_eeg_test.npy'."""
    if _settings['local_root']:
        return os.path.join(_settings['local_root'], rel)
    if not _settings['repo_id']:
        raise RuntimeError("Set the dataset repo: eeg_data.configure(repo_id='org/name') or the HF_REPO_ID environment variable")
    from huggingface_hub import hf_hub_download
    os.makedirs(_settings['work_dir'], exist_ok=True)
    return hf_hub_download(_settings['repo_id'], rel, repo_type='dataset', token=_token(), local_dir=_settings['work_dir'])


def release(path):
    """Delete a downloaded file to save disk (a local copy is never deleted)."""
    if not _settings['local_root'] and os.path.exists(path):
        os.remove(path)


# restricted unpickler: the .npy files are pickled dicts, only NumPy types are allowed
_OK = {('numpy', 'ndarray'), ('numpy', 'dtype'),
       ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
       ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
       ('builtins', 'dict'), ('builtins', 'list')}


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _OK:
            return getattr(__import__(module, fromlist=[name]), name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')


def safe_load(path):
    with open(path, 'rb') as f:
        v = np.lib.format.read_magic(f)
        np.lib.format._read_array_header(f, v)
        obj = _U(f).load()
    return obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj


# ------------------------------ config + split bookkeeping ------------------------------
@lru_cache(maxsize=1)
def state():
    """Loads the config files and the image metadata once."""
    cdir = _settings['config_dir']
    try:
        cfg = json.load(open(os.path.join(cdir, 'preproc_stats.json')))
        split = json.load(open(os.path.join(cdir, 'split_concepts.json')))
    except FileNotFoundError as e:
        raise FileNotFoundError(f'{e}. Put preproc_stats.json and split_concepts.json into "{cdir}" (or call eeg_data.configure(config_dir=...))')
    assert cfg['source'] == '17ch', 'the statistics were fitted on the 17-channel files; refit them for another channel set'
    ts = np.arange(100) * 0.01 - 0.2                                   # -0.20 ... 0.79 s, stimulus onset at 0
    lo, hi = cfg['crop_ms']
    crop = (ts >= lo / 1000 - 1e-9) & (ts <= hi / 1000 + 1e-9)
    assert int(crop.sum()) == cfg['n_time_points']
    meta = safe_load(get_file('images/image_metadata.npy'))
    tr_conc = np.array(meta['train_img_concepts'])
    assert list(tr_conc) == sorted(tr_conc), 'EEG rows are expected to be sorted by concept'
    is_val = np.isin(tr_conc, split['validation_concepts'])
    assert is_val.sum() == 10 * len(split['validation_concepts']) and len(is_val) == 16540
    common = dict(source=cfg['source'], channels=cfg['channels'], crop_ms=cfg['crop_ms'], n_time_points=cfg['n_time_points'],
                  clip_k=cfg['clip_k'], own=cfg['own'], validation_concepts=split['validation_concepts'])
    return SimpleNamespace(
        cfg=cfg, split=split, ts=ts, crop=crop, clip_k=float(cfg['clip_k']), channels=cfg['channels'],
        train_concepts=tr_conc, is_val=is_val, train_idx=np.flatnonzero(~is_val), val_idx=np.flatnonzero(is_val),
        train_files=list(zip(tr_conc, meta['train_img_files'])),          # (concept folder, file name) per EEG row
        test_files=list(zip(meta['test_img_concepts'], meta['test_img_files'])),
        config_id=hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()[:12],
        subjects=list(cfg['own']))


def info():
    """Short summary of the loaded configuration (config_id must be identical for everyone in the team)."""
    s = state()
    return dict(config_id=s.config_id, subjects=s.subjects, n_channels=len(s.channels), window_ms=s.cfg['crop_ms'],
                n_time_points=int(s.crop.sum()), clip_k=s.clip_k, n_train_images=len(s.train_idx), n_val_images=len(s.val_idx))


def train_files():
    """(concept folder, file name) of the 16,540 training images, in EEG row order (use it to order image embeddings)."""
    return state().train_files


def test_files():
    """(concept folder, file name) of the 200 test images, in EEG row order."""
    return state().test_files


# ------------------------------ preprocessing ------------------------------
def stats_of(d):
    return dict(median=np.asarray(d['median'], np.float32), scale=np.asarray(d['scale'], np.float32))


def preprocess(x, stats, k=None, crop=True):
    """x: (..., 17, 100) raw trials -> (..., 17, 60): crop, remove the median, divide by the robust scale, clip to +-k."""
    st = state()
    k = st.clip_k if k is None else k
    if crop:
        x = x[..., st.crop]
    z = (x - stats['median'][:, None]) / stats['scale'][:, None]
    return np.clip(z, -k, k).astype(np.float32)


def load_eeg(s, split):
    """Raw EEG of subject s: split='training' -> (16540, 4, 17, 100), 'test' -> (200, 80, 17, 100)."""
    p = get_file(f'eeg/17ch/{s}/preprocessed_eeg_{split}.npy')
    T = safe_load(p)
    release(p)
    X = T['preprocessed_eeg_data']
    assert T['ch_names'] == state().channels and X.shape[2:] == (17, 100)
    return X


def _prep_rows(Y, rows, stats, avg_reps, chunk=2000):
    out = []
    for a in range(0, len(rows), chunk):
        z = preprocess(Y[rows[a:a + chunk]], stats)
        out.append(z.mean(1) if avg_reps else z)
    return np.concatenate(out)


def subject_data(s, stats, parts=('train', 'val'), avg_reps=True):
    """Preprocess subject s with the given statistics. parts: any of 'train', 'val', 'test'."""
    st = state()
    d = {}
    if 'train' in parts or 'val' in parts:
        Y = load_eeg(s, 'training')
        assert Y.shape == (16540, 4, 17, 100), Y.shape
        if 'train' in parts:
            d['X_train'] = _prep_rows(Y, st.train_idx, stats, avg_reps)
            d['idx_train'] = st.train_idx
        if 'val' in parts:
            d['X_val'] = _prep_rows(Y, st.val_idx, stats, avg_reps)
            d['idx_val'] = st.val_idx
        del Y
        gc.collect()
    if 'test' in parts:
        X = load_eeg(s, 'test')
        assert X.shape == (200, 80, 17, 100), X.shape
        d['X_test'] = _prep_rows(X, np.arange(200), stats, avg_reps)
        del X
    return d
