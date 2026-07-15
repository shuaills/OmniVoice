import numpy as np

from data_audit.emilia_waveform_tail_audit import estimate_tail_seconds


def _tone(seconds: float, sample_rate: int, frequency: float, amplitude: float):
    time = np.arange(int(seconds * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * time)


def test_estimates_one_second_quiet_tail():
    sample_rate = 16000
    audio = np.concatenate(
        [_tone(2.0, sample_rate, 440.0, 0.2), np.zeros(sample_rate)]
    )
    result = estimate_tail_seconds(audio, sample_rate)
    assert 0.85 <= result["tail_s"] <= 1.15


def test_isolated_tail_click_does_not_move_speech_end():
    sample_rate = 16000
    audio = np.concatenate(
        [_tone(2.0, sample_rate, 440.0, 0.2), np.zeros(sample_rate)]
    )
    audio[-800] = 1.0
    result = estimate_tail_seconds(audio, sample_rate)
    assert result["tail_s"] >= 0.8


def test_low_frequency_hum_is_suppressed_by_preemphasis():
    sample_rate = 16000
    audio = np.concatenate(
        [
            _tone(2.0, sample_rate, 440.0, 0.2),
            _tone(1.0, sample_rate, 150.0, 0.005),
        ]
    )
    result = estimate_tail_seconds(audio, sample_rate)
    assert result["tail_s"] >= 0.8
