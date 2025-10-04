from __future__ import annotations

from typing import Dict

import dash
from dash import Input, Output, dcc, html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import sqlite3

from data.datasets import SequenceDataset


def _load_predictions(db_path: str, table_name: str = "model_b_inference") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()
    return df


def _load_dataset(dataset_path: str) -> SequenceDataset:
    return SequenceDataset.load(dataset_path)


def run_visual_tool(config: Dict) -> None:
    db_path = config.get("db_path", "artifacts/predictions.db")
    dataset_path = config.get("dataset_path", "artifacts/test_dataset.npz")
    forecast_length = config.get("forecast_length")

    df_all = _load_predictions(db_path)
    test_dataset = _load_dataset(dataset_path)
    if forecast_length is None:
        forecast_length = test_dataset.targets.shape[1]

    y_cols = [f"y_{i+1}" for i in range(forecast_length)]
    p_cols = [f"p_{i+1}" for i in range(forecast_length)]
    unique_groups = sorted(df_all["group_id"].unique().tolist())

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.H3("輸入資料與預測值對照圖"),
            html.Label("選擇 group："),
            dcc.Dropdown(
                id="group-dropdown",
                options=[{"label": str(g), "value": g} for g in unique_groups],
                value=unique_groups[0] if unique_groups else None,
                clearable=False,
            ),
            html.Label("選擇 sample_idx："),
            dcc.Dropdown(id="sample-dropdown", options=[], value=None, clearable=False),
            dcc.Graph(id="prediction-graph"),
        ]
    )

    @app.callback(Output("sample-dropdown", "options"), Output("sample-dropdown", "value"), Input("group-dropdown", "value"))
    def update_sample_options(selected_group):
        if selected_group is None:
            return [], None
        df_group = df_all[df_all["group_id"] == selected_group]
        candidate_samples = sorted(df_group["sample_idx"].unique().tolist())
        if not candidate_samples:
            return [], None
        options = [{"label": str(s), "value": s} for s in candidate_samples]
        return options, candidate_samples[0]

    @app.callback(
        Output("prediction-graph", "figure"),
        Input("group-dropdown", "value"),
        Input("sample-dropdown", "value"),
    )
    def update_graph(selected_group, selected_sample):
        if selected_group is None or selected_sample is None:
            return go.Figure()

        df_sel = df_all[(df_all["group_id"] == selected_group) & (df_all["sample_idx"] == selected_sample)]
        if df_sel.empty:
            return go.Figure()

        valid_length = int(df_sel["valid_length"].values[0])
        true_values = df_sel[y_cols].values[0][:valid_length]
        pred_values = df_sel[p_cols].values[0][:valid_length]
        x_forecast = list(range(1, valid_length + 1))

        sample_idx = int(selected_sample)
        X_sample = test_dataset.inputs[sample_idx]
        mask_sample = test_dataset.input_mask[sample_idx]
        effective_input_length = int(np.sum(mask_sample))
        input_values = X_sample[-effective_input_length:, 3]
        x_input = list(range(-effective_input_length + 1, 1))

        trace_input = go.Scatter(
            x=x_input,
            y=input_values,
            mode="lines+markers",
            name="輸入資料 (Close)",
            line=dict(color="blue"),
        )
        trace_true = go.Scatter(
            x=x_forecast,
            y=true_values,
            mode="lines+markers",
            name="實際值",
            line=dict(color="green"),
        )
        trace_pred = go.Scatter(
            x=x_forecast,
            y=pred_values,
            mode="lines+markers",
            name="模型預測",
            line=dict(color="red"),
        )

        shapes = [
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=min(x_input),
                y0=0,
                x1=0,
                y1=1,
                fillcolor="lightblue",
                opacity=0.2,
                layer="below",
                line_width=0,
            ),
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=0,
                y0=0,
                x1=max(x_forecast) if x_forecast else 1,
                y1=1,
                fillcolor="lightgreen",
                opacity=0.2,
                layer="below",
                line_width=0,
            ),
        ]

        fig = go.Figure(data=[trace_input, trace_true, trace_pred])
        fig.update_layout(
            title=(
                f"Group {selected_group} / Sample {selected_sample} "
                f"(有效輸入長度={effective_input_length}, 有效預測長度={valid_length})"
            ),
            xaxis_title="時間 (負數：輸入，正數：預測)",
            yaxis_title="數值",
            shapes=shapes,
        )
        return fig

    app.run_server(debug=True, use_reloader=False)
