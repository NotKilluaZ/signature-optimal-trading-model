from __future__ import annotations
import numpy as np
from src.sigstop.features.augmentation import make_time_channel
from src.sigstop.features.feature_builder import (
    build_batched_feature_tensor,
    build_feature_tensor,
)
from src.sigstop.features.scaling import fit_zscore_scaler
from src.sigstop.features.signature_extractor import expected_signature_dim
from src.sigstop.features.validation import validate_prefix_feature_tensor


def test_batched_feature_builder_matches_single_path_pipeline() -> None:
    formation_spread = np.array([10.0, 12.0, 14.0, 16.0], dtype = np.float64)
    synthetic_paths = np.array(
        [
            [10.0, 11.0, 12.0, 13.0],
            [12.0, 13.0, 14.0, 15.0],
        ],
        dtype = np.float64,
    )
    scaler = fit_zscore_scaler(formation_spread)

    batched = build_batched_feature_tensor(
        synthetic_paths,
        scaler,
        depth = 2,
        min_prefix = 2,
        scalar_term = True,
        dtype = np.float32,
        library = "esig",
        device = "cpu",
    )
    single_0 = build_feature_tensor(
        synthetic_paths[0],
        scaler,
        depth = 2,
        min_prefix = 2,
        scalar_term = True,
        dtype = np.float32,
        library = "esig",
        device = "cpu",
    )
    single_1 = build_feature_tensor(
        synthetic_paths[1],
        scaler,
        depth = 2,
        min_prefix = 2,
        scalar_term = True,
        dtype = np.float32,
        library = "esig",
        device = "cpu",
    )

    np.testing.assert_allclose(batched.scaled_spreads[0], single_0.scaled_spread)
    np.testing.assert_allclose(batched.scaled_spreads[1], single_1.scaled_spread)
    np.testing.assert_allclose(batched.augmented_paths[0], single_0.augmented_path)
    np.testing.assert_allclose(batched.augmented_paths[1], single_1.augmented_path)
    np.testing.assert_allclose(batched.features[0], single_0.features)
    np.testing.assert_allclose(batched.features[1], single_1.features)
    np.testing.assert_array_equal(batched.prefix_ends, single_0.prefix_ends)


def test_batched_feature_builder_uses_shared_time_channel_and_expected_shape() -> None:
    formation_spread = np.array([1.0, 2.0, 3.0, 4.0], dtype = np.float64)
    synthetic_paths = np.array(
        [
            [1.0, 1.5, 2.0, 2.5, 3.0],
            [2.0, 1.0, 0.5, 1.5, 2.5],
            [3.0, 2.5, 2.0, 1.5, 1.0],
        ],
        dtype = np.float64,
    )
    scaler = fit_zscore_scaler(formation_spread)

    result = build_batched_feature_tensor(
        synthetic_paths,
        scaler,
        depth = 3,
        min_prefix = 2,
        scalar_term = False,
        dtype = np.float32,
        library = "esig",
        device = "cpu",
    )

    expected_dim = expected_signature_dim(channels = 2, depth = 3, scalar_term = False)
    expected_time = make_time_channel(5, representation = "index_0_1", dtype = np.float32)

    assert result.features.shape == (3, 4, expected_dim)
    np.testing.assert_array_equal(result.prefix_ends, np.array([1, 2, 3, 4], dtype = np.int32))
    np.testing.assert_allclose(
        result.augmented_paths[:, :, 0],
        np.broadcast_to(expected_time, (3, 5)),
    )
    validate_prefix_feature_tensor(
        result.features,
        channels = 2,
        depth = 3,
        scalar_term = False,
        expected_rank = 3,
        name = "synthetic_batch_features",
    )


def test_batched_feature_builder_preserves_loss_alignment_contract() -> None:
    formation_spread = np.array([5.0, 6.0, 7.0, 8.0], dtype = np.float64)
    synthetic_paths = np.array(
        [
            [5.0, 5.2, 5.1, 5.3, 5.4, 5.6],
            [6.0, 5.8, 5.7, 5.9, 6.1, 6.2],
        ],
        dtype = np.float64,
    )
    scaler = fit_zscore_scaler(formation_spread)

    result = build_batched_feature_tensor(
        synthetic_paths,
        scaler,
        depth = 2,
        min_prefix = 2,
        scalar_term = True,
        dtype = np.float32,
        library = "esig",
        device = "cpu",
    )

    assert result.features.shape[1] == synthetic_paths.shape[1] - 1
    assert len(result.prefix_ends) == synthetic_paths.shape[1] - 1
    assert result.prefix_ends[0] == 1
    assert result.prefix_ends[-1] == synthetic_paths.shape[1] - 1
