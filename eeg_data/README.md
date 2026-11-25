# eeg_data: shared THINGS-EEG2 loaders

One implementation of the split and the preprocessing for the whole team.

| File | What it contains |
|---|---|
| `common.py` | data access (Hugging Face or a local copy), config files, `preprocess` (shared by both splits) |
| `within_subject.py` | `within_subject(s)`: train, validation and test on the same person |
| `loso.py` | `loso(s)`: leave-one-subject-out (7 training, 2 validation, 1 test person) |


The numbers (165 validation concepts, per-subject statistics, clip K, window) are read from `../configs/split_concepts.json` and `../configs/preproc_stats.json`. They come from `notebooks/split_and_preprocessing.ipynb`. Do not edit them by hand.
`eeg_data.info()['config_id']` must be the same for everybody in the team.

## Use in Colab
```python
!git clone <repo url>          # (private repo: use a GitHub token in the url)
!pip install -q huggingface_hub
import sys; sys.path.append('/content/<repo folder>')
import eeg_data
eeg_data.configure(repo_id='ORG/DATASET_NAME')       # HF token is taken from the Colab Secret HF_TOKEN
print(eeg_data.info())

from eeg_data.within_subject import within_subject
d = within_subject('sub-03')       # X_train (14890, 17, 60), X_val (1650, 17, 60), idx_train, idx_val

from eeg_data.loso import loso
L = loso('sub-03')                 # 7 training subjects, 2 validation subjects (new-subject normalisation 'B')
L = loso('sub-03', new_subject_norm='A')     # ablation
d = within_subject('sub-03', include_test=True)   # final evaluation only
```
`idx_train` / `idx_val` are row numbers of the 16,540 training images in the order of `eeg_data.train_files()`; test rows follow `eeg_data.test_files()`. Use these orders for the image embeddings.

## Notes
- Repetitions are averaged after clipping (`avg_reps=False` keeps them). Test images are averaged over 80 repetitions, training images over 4.
- Test data are returned only with `include_test=True`.
- The statistics were fitted on the 17-channel files; for another channel set run the preprocessing notebook again and replace `configs/`.
- Local copy instead of downloading: `eeg_data.configure(local_root='path/to/dataset')` (folder with `eeg/17ch/...` and `images/...`).
