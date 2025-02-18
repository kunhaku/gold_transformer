import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import sqlite3
import pandas as pd
import numpy as np

# === 1) 從 SQLite 資料庫讀取推論結果 ===
db_path = "predictions.db"
conn = sqlite3.connect(db_path)
df_all = pd.read_sql_query("SELECT * FROM model_b_inference", conn)
conn.close()

# df_all 應含欄位:
#  - id, group_id, sample_idx, valid_length
#  - y_1..y_n, p_1..p_n
forecast_length = 10
y_cols = [f"y_{i+1}" for i in range(forecast_length)]
p_cols = [f"p_{i+1}" for i in range(forecast_length)]

# === 2) 從原始測試資料中讀取輸入序列資訊 ===
# 這裡假設 test_raw.npz 存有原始的 X_data 與 mask_data
test_raw = np.load("test_raw.npz")
X_raw = test_raw["X_data"]       # shape: (num_samples, max_input_length, input_dim)
mask_raw = test_raw["mask_data"] # shape: (num_samples, max_input_length, 1)
# 注意：根據你的說法，mt5 data db 中的 close 是第五欄，
# 因此這邊我們取 X_raw 的第 5 列 (index=4) 當作 close 價格

# === 3) 確認可用的 group 與樣本清單 ===
unique_groups = sorted(df_all["group_id"].unique().tolist())

def get_samples_by_group(g):
    df_group = df_all[df_all["group_id"] == g].copy()
    chosen_idx = sorted(df_group["sample_idx"].unique().tolist())
    return chosen_idx

# === 4) 建立 Dash 應用，配置基礎介面 ===
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("輸入資料與預測值對照圖"),
    html.Label("選擇 group："),
    dcc.Dropdown(
        id="group-dropdown",
        options=[{"label": str(g), "value": g} for g in unique_groups],
        value=unique_groups[0],
        clearable=False
    ),
    html.Label("選擇 sample_idx："),
    dcc.Dropdown(
        id="sample-dropdown",
        options=[],
        value=None,
        clearable=False
    ),
    dcc.Graph(id="prediction-graph")
])

# === 5) Callback：根據選到的 group 更新 sample-dropdown 的選項 ===
@app.callback(
    Output("sample-dropdown", "options"),
    Output("sample-dropdown", "value"),
    Input("group-dropdown", "value")
)
def update_sample_options(selected_group):
    if selected_group is None:
        return [], None
    candidate_samples = get_samples_by_group(selected_group)
    if not candidate_samples:
        return [], None
    options = [{"label": str(s), "value": s} for s in candidate_samples]
    return options, candidate_samples[0]

# === 6) Callback：根據 group 與 sample_idx 繪圖 ===
@app.callback(
    Output("prediction-graph", "figure"),
    Input("group-dropdown", "value"),
    Input("sample-dropdown", "value")
)
def update_graph(selected_group, selected_sample):
    if (selected_group is None) or (selected_sample is None):
        return go.Figure()

    # 從 DB 中取出預測結果資料
    df_sel = df_all[(df_all["group_id"] == selected_group) &
                    (df_all["sample_idx"] == selected_sample)]
    if df_sel.empty:
        return go.Figure()

    # 讀取有效預測長度
    valid_length = int(df_sel["valid_length"].values[0])
    # 讀取真實預測與模型預測（forecast部分）
    true_values = df_sel[y_cols].values[0]
    pred_values = df_sel[p_cols].values[0]
    if valid_length > 0:
        true_values = true_values[:valid_length]
        pred_values = pred_values[:valid_length]
    else:
        return go.Figure()
    x_forecast = list(range(1, valid_length + 1))

    # 從原始資料中取得該筆的輸入序列
    sample_idx = int(selected_sample)
    X_sample = X_raw[sample_idx]       # shape: (max_input_length, input_dim)
    mask_sample = mask_raw[sample_idx] # shape: (max_input_length, 1)
    effective_input_length = int(np.sum(mask_sample))
    # 取 close 價格，這裡 close 為第 5 欄，即 index=4
    input_values = X_sample[-effective_input_length:, 3]

    x_input = list(range(-effective_input_length+1, 1))

    # 建立各個 trace
    trace_input = go.Scatter(
        x=x_input,
        y=input_values,
        mode="lines+markers",
        name="輸入資料 (Close)",
        line=dict(color="blue")
    )
    trace_true = go.Scatter(
        x=x_forecast,
        y=true_values,
        mode="lines+markers",
        name="實際值",
        line=dict(color="green")
    )
    trace_pred = go.Scatter(
        x=x_forecast,
        y=pred_values,
        mode="lines+markers",
        name="模型預測",
        line=dict(color="red")
    )

    # 設定背景區塊：左側為輸入區，右側為預測區
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
            line_width=0
        ),
        dict(
            type="rect",
            xref="x",
            yref="paper",
            x0=0,
            y0=0,
            x1=max(x_forecast),
            y1=1,
            fillcolor="lightgreen",
            opacity=0.2,
            layer="below",
            line_width=0
        )
    ]

    fig = go.Figure(data=[trace_input, trace_true, trace_pred])
    fig.update_layout(
        title=f"Group {selected_group} / Sample {selected_sample} (有效輸入長度={effective_input_length}, 有效預測長度={valid_length})",
        xaxis_title="時間 (負數：輸入，正數：預測)",
        yaxis_title="數值",
        shapes=shapes
    )
    return fig

def run_visual_tool(config):
    app.run_server(debug=True)

if __name__ == "__main__":
    run_visual_tool({})
