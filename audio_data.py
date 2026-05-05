"""
Data loading and preprocessing for the Shader et al. fNIRS auditory speech dataset.

Pipeline (from the paper, following Shader et al. [18]):
  raw intensity -> optical density -> SCI < 0.8 channel rejection ->
  TDDR motion correction -> short-channel regression -> Beer-Lambert (HbO/HbR) ->
  bandpass 0.02-0.4 Hz -> negative-correlation enhancement -> epoching with 100 uM rejection
"""
import mne
import mne_nirs
import numpy as np
from itertools import compress
from mne.preprocessing.nirs import (
    optical_density,
    temporal_derivative_distribution_repair,
)
from mne_bids import BIDSPath, get_entity_vals


def epoch_preprocessing(subject_index, t_min, t_max, l_freq=0.02, h_freq=0.4):
    """Preprocess one subject and return haemoglobin epochs.

    Args:
        subject_index: 0..7
        t_min, t_max: epoch window in seconds (paper uses 0.0 to 18.0)
        l_freq, h_freq: bandpass cutoffs in Hz (paper uses 0.02 / 0.4)

    Returns:
        haemo: continuous haemoglobin recording
        epochs: mne.Epochs containing only the "Audio" and "Control" trials
    """
    root = mne_nirs.datasets.audio_or_visual_speech.data_path()
    subject = get_entity_vals(root, "subject")[subject_index]

    dataset = BIDSPath(
        root=root,
        suffix="nirs",
        extension=".snirf",
        subject=subject,
        task="AudioVisualBroadVsRestricted",
        datatype="nirs",
        session="01",
    )

    raw_intensity = mne.io.read_raw_snirf(dataset.fpath)
    raw_intensity.annotations.rename(
        {"1.0": "Audio", "2.0": "Video", "3.0": "Control", "15.0": "Ends"}
    )

    raw_od = optical_density(raw_intensity)

    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.8))
    raw_od = raw_od.drop_channels(raw_od.info["bads"])

    corrected_tddr = temporal_derivative_distribution_repair(raw_od)
    od_corrected = mne_nirs.signal_enhancement.short_channel_regression(corrected_tddr)

    haemo = mne.preprocessing.nirs.beer_lambert_law(od_corrected, ppf=6)
    haemo = mne_nirs.channels.get_long_channels(haemo)
    haemo = haemo.filter(l_freq, h_freq)
    haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(haemo)

    events, event_dict = mne.events_from_annotations(haemo)
    epochs = mne.Epochs(
        haemo,
        events,
        event_id=event_dict,
        tmin=t_min,
        tmax=t_max,
        reject=dict(hbo=100e-6, hbr=100e-6),
        reject_by_annotation=True,
        proj=False,
        baseline=None,
        detrend=None,
        preload=True,
        verbose=True,
    )

    epochs = epochs[["Audio", "Control"]]
    return haemo, epochs


def final_audio_data(subject_index, t_min, t_max, chroma=None, l_freq=0.02, h_freq=0.4):
    """Return classification-ready features for one subject.

    Args:
        subject_index: 0..7
        t_min, t_max: epoch window in seconds
        chroma: None (return both HbO and HbR), 'hbo', or 'hbr'

    Returns:
        X: (n_trials, n_channels, n_times) float array
        Y: (n_trials,) int array with -1 for Audio, +1 for Control (silence)
        epochs: the underlying mne.Epochs (useful for channel info)
    """
    _, epochs = epoch_preprocessing(subject_index, t_min, t_max, l_freq, h_freq)

    if chroma is not None:
        epochs.pick(chroma)

    X = epochs.get_data()
    Y = epochs.events[:, 2]
    label_map = {1: -1, 2: 1}
    Y = np.array([label_map[y] for y in Y])

    return X, Y, epochs
