import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook
from run_analysis import build_metrics

from src.demo_data import generate_demo_eeg
from src.dsp_pipeline import (
    FS_ORIGINAL,
    FS_TARGET,
    band_power_features,
    comparison_metrics,
    load_eeg_xlsx,
    RESAMPLE_TAPS,
    design_resample_fir,
    epoch_psd,
    filter_response_metrics,
    iwmf_features,
    normalize_psd,
    resample_direct_reference,
    resample_polyphase,
    segment_epochs,
    spectral_edge_frequency_features,
    spectral_entropy_features,
    spectral_preservation_metrics,
    theoretical_complexity,
)


class TestEE6041Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_demo_eeg(fs=FS_ORIGINAL, seconds=60.0)
        cls.taps = design_resample_fir()
        cls.direct = resample_direct_reference(cls.data, cls.taps)
        cls.poly = resample_polyphase(cls.data, cls.taps)
        cls.raw_epochs, _ = segment_epochs(cls.data, FS_ORIGINAL)
        cls.poly_epochs, _ = segment_epochs(cls.poly, FS_TARGET)

    def test_expected_output_length(self):
        self.assertEqual(len(self.poly), 1920)
        self.assertEqual(len(self.direct), 1904)

    def test_arbitrary_lengths_and_boundary_impulses(self):
        for size in (260, 501, 799, 4001):
            for position in (0, size // 2, size - 1):
                with self.subTest(size=size, position=position):
                    data = np.zeros(size)
                    data[position] = 1
                    direct = resample_direct_reference(data, self.taps)
                    poly = resample_polyphase(data, self.taps)
                    self.assertEqual(len(poly), (size * 8 + 124) // 125)
                    np.testing.assert_allclose(poly[:len(direct)], direct, atol=1e-14, rtol=0)

    def test_dc_gain(self):
        poly = resample_polyphase(np.ones(4000), self.taps)
        # Allow the 60 dB design's 0.001 linear-amplitude tolerance.
        np.testing.assert_allclose(poly[20:-20], 1, atol=1e-3, rtol=0)

    def test_partial_epoch_and_expected_length(self):
        with patch("run_analysis.benchmark_resamplers", return_value={}):
            for size in (4001, 5999):
                metrics, series = build_metrics(self.data[:size])
                self.assertEqual(metrics["resampling"]["expected_output_samples"], (size * 8 + 124) // 125)
                self.assertEqual(len(series["raw_times"]), len(series["poly_times"]))

    def test_invalid_recordings(self):
        for data in ([], np.zeros(3999), np.full(4000, np.inf), np.zeros((4000, 2))):
            with self.assertRaises(ValueError):
                build_metrics(data)

    def test_xlsx_first_column_validation(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "input.xlsx"
            for values in ([], ["EEG", 1], [1, None, 2], [1, "inf"]):
                wb = Workbook()
                for value in values:
                    wb.active.append([value])
                wb.save(path)
                wb.close()
                with self.assertRaises(ValueError):
                    load_eeg_xlsx(path)
            wb = Workbook()
            wb.active.append([1, "ignored"])
            wb.active.append([2, "ignored"])
            wb.save(path)
            wb.close()
            np.testing.assert_array_equal(load_eeg_xlsx(path), [1, 2])

    def test_low_amplitude_and_constant_correlations(self):
        values = np.array([1, 2, 4, 8]) * 1e-12
        self.assertAlmostEqual(comparison_metrics(values, -values)["correlation"], -1)
        self.assertTrue(np.isnan(comparison_metrics(np.ones(3), np.ones(3))["correlation"]))

    def test_known_tone_power_and_frequency(self):
        for fs in (FS_ORIGINAL, FS_TARGET):
            t = np.arange(int(8 * fs)) / fs
            epochs = (2 * np.sin(2 * np.pi * 6 * t))[None, :]
            freqs, psd = epoch_psd(epochs, fs)
            self.assertAlmostEqual(float(psd.sum() * (freqs[1] - freqs[0])), 2, places=10)
            self.assertAlmostEqual(band_power_features(epochs, fs)["mid_4_8_hz"][0], 2, places=10)
            self.assertAlmostEqual(iwmf_features(epochs, fs)[0], 6, places=10)
            self.assertAlmostEqual(spectral_edge_frequency_features(epochs, fs)[0], 6.125)
            probabilities = np.array([1 / 6, 2 / 3, 1 / 6])
            expected_entropy = -np.sum(probabilities * np.log(probabilities)) / np.log(97)
            self.assertAlmostEqual(spectral_entropy_features(epochs, fs)[0], expected_entropy, places=10)

    def test_zero_power_is_not_preservation(self):
        epochs = np.zeros((2, 256))
        with self.assertRaises(ValueError):
            spectral_preservation_metrics(epochs, 32, epochs, 32)

    def test_demo_is_deterministic(self):
        np.testing.assert_array_equal(self.data, generate_demo_eeg())

    def test_polyphase_matches_direct_reference(self):
        n = len(self.direct)
        np.testing.assert_allclose(self.poly[:n], self.direct, rtol=0.0, atol=1e-10)

    def test_filter_design_achieved_response(self):
        stats = filter_response_metrics(self.taps)
        self.assertEqual(stats["num_taps"], RESAMPLE_TAPS)
        self.assertEqual(RESAMPLE_TAPS, 4145)
        self.assertAlmostEqual(self.taps.sum(), 1.0, places=12)
        np.testing.assert_allclose(self.taps, self.taps[::-1], atol=1e-15)
        self.assertGreaterEqual(stats["worst_case_passband_deviation_db"], abs(stats["passband_edge_gain_db"]))
        self.assertLess(abs(stats["passband_edge_gain_db"]), 0.05)
        self.assertGreater(stats["minimum_stopband_attenuation_db"], 59.8)

    def test_epoch_count_and_resolution(self):
        self.assertEqual(len(self.raw_epochs), 14)
        self.assertEqual(len(self.poly_epochs), 14)
        self.assertAlmostEqual(FS_ORIGINAL / self.raw_epochs.shape[1], 0.125)
        self.assertAlmostEqual(FS_TARGET / self.poly_epochs.shape[1], 0.125)

    def test_psd_frequency_grid_alignment(self):
        raw_freqs, _ = epoch_psd(self.raw_epochs, FS_ORIGINAL)
        poly_freqs, _ = epoch_psd(self.poly_epochs, FS_TARGET)
        np.testing.assert_allclose(raw_freqs, poly_freqs, rtol=0.0, atol=1e-12)
        self.assertEqual(len(raw_freqs), 97)
        self.assertAlmostEqual(raw_freqs[1] - raw_freqs[0], 0.125)

    def test_normalized_psd_sums_to_one(self):
        _, psd = epoch_psd(self.poly_epochs, FS_TARGET)
        normalized = normalize_psd(psd)
        np.testing.assert_allclose(normalized.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)

    def test_iwmf_shapes_match(self):
        raw_feature = iwmf_features(self.raw_epochs, FS_ORIGINAL)
        poly_feature = iwmf_features(self.poly_epochs, FS_TARGET)
        self.assertEqual(raw_feature.shape, poly_feature.shape)
        self.assertTrue(np.isfinite(raw_feature).all())
        self.assertTrue(np.isfinite(poly_feature).all())

    def test_spectral_entropy_and_sef95_are_valid(self):
        entropy = spectral_entropy_features(self.poly_epochs, FS_TARGET)
        sef95 = spectral_edge_frequency_features(self.poly_epochs, FS_TARGET)
        self.assertTrue(np.all((entropy >= 0.0) & (entropy <= 1.0)))
        self.assertTrue(np.all((sef95 >= 0.5) & (sef95 <= 12.5)))

    def test_identity_spectral_validation(self):
        metrics = spectral_preservation_metrics(
            self.raw_epochs,
            FS_ORIGINAL,
            self.raw_epochs,
            FS_ORIGINAL,
        )
        self.assertAlmostEqual(metrics["psd_shape"]["correlation"]["mean"], 1.0, places=12)
        self.assertAlmostEqual(metrics["psd_shape"]["normalized_psd_rmse"]["max"], 0.0, places=12)
        for band in metrics["band_power_relative_error_percent"].values():
            self.assertAlmostEqual(band["max"], 0.0, places=12)
        self.assertAlmostEqual(metrics["spectral_entropy"]["rmse"], 0.0, places=12)
        self.assertAlmostEqual(metrics["sef95"]["rmse"], 0.0, places=12)

    def test_resampled_spectral_validation_is_finite(self):
        metrics = spectral_preservation_metrics(
            self.raw_epochs,
            FS_ORIGINAL,
            self.poly_epochs,
            FS_TARGET,
        )
        self.assertTrue(np.isfinite(metrics["psd_shape"]["correlation"]["mean"]))
        self.assertTrue(np.isfinite(metrics["spectral_entropy"]["rmse"]))
        self.assertTrue(np.isfinite(metrics["sef95"]["rmse"]))

    def test_theoretical_complexity_reduction(self):
        stats = theoretical_complexity(len(self.data))
        self.assertEqual(stats["direct_form_macs"], len(self.data) * 8 * RESAMPLE_TAPS)
        self.assertAlmostEqual(
            stats["polyphase_core_macs_approx"],
            len(self.data) * RESAMPLE_TAPS / 125,
        )
        self.assertAlmostEqual(stats["theoretical_reduction_factor"], 1000.0)


if __name__ == "__main__":
    unittest.main()
