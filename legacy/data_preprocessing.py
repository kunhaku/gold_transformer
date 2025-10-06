"""Legacy helpers delegating to the new data processing modules."""

from __future__ import annotations

from configs import DataConfig
from data.datasets import SequenceDataset, build_sequence_dataset, split_train_test
from data.features import build_feature_frame
from data.ingest import load_mt5_data as _load_mt5_data
from data.windowing import create_sliding_windows


def load_mt5_data(db_path: str | None = None, table_name: str | None = None):
    config = DataConfig()
    if db_path:
        config.db_path = db_path
    if table_name:
        config.table_name = table_name
    raw_df = _load_mt5_data(config)
    return build_feature_frame(raw_df)


def prepare_data_and_split(
    df=None,
    config: DataConfig | dict | None = None,
):
    data_config = config if isinstance(config, DataConfig) else DataConfig()

    if df is None:
        dataset = build_sequence_dataset(data_config)
    else:
        if isinstance(df, SequenceDataset):
            dataset = df
        else:
            feature_matrix = df[data_config.feature_columns()].values
            X, y, mask, y_mask, group_ids = create_sliding_windows(feature_matrix, data_config)
            dataset = SequenceDataset(X, y, mask, y_mask, group_ids)

    train_dataset, test_dataset = split_train_test(dataset, data_config)
    return (
        train_dataset.inputs,
        train_dataset.targets,
        train_dataset.input_mask,
        train_dataset.target_mask,
        train_dataset.group_ids,
    ), (
        test_dataset.inputs,
        test_dataset.targets,
        test_dataset.input_mask,
        test_dataset.target_mask,
        test_dataset.group_ids,
    )


def save_data_numpy(*args, **kwargs):
    raise NotImplementedError("Use SequenceDataset.save from data.datasets instead.")
