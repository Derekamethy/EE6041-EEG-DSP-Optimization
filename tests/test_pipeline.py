import unittest

import numpy as np

from src.demo_data import generate_demo_eeg
from src.dsp_pipeline import (
    FS_ORIGINAL,
    FS_TARGET,
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
        self.assertGreater(len(self.direct), 0)

    def test_polyphase_matches_direct_reference(self):
        n = len(self.direct)
        np.testing.assert_allclose(self.poly[:n], self.direct, rtol=0.0, atol=1e-10)

    def test_filter_design_meets_target_edges(self):
        stats = filter_response_metrics(self.taps)
        self.assertEqual(stats["num_taps"], RESAMPLE_TAPS)
        self.assertLess(abs(stats["passband_edge_gain_db"]), 0.05)
        self.assertGreater(stats["minimum_stopband_attenuation_db"], 59.0)

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