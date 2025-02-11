import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import sqlite3
import pandas as pd

# === 1) 從 SQLite 資料庫讀取推論結果 ===
db_path = "predictions.db"
conn = sqlite3.connect(db_path)
df_all = pd.read_sql_query("SELECT * FROM model_b_inference", conn)
conn.close()

# df_all 應含欄位:
#  - id, group_id, sample_idx, valid_length
#  - y_1..y_n, p_1..p_n

# 這裡假設 forecast_length=10，若不同可自行偵測
forecast_length = 10
y_cols = [f"y_{i+1}" for i in range(forecast_length)]
p_cols = [f"p_{i+1}" for i in range(forecast_length)]

# === 2) 確認可用的 group 與樣本清單 ===
unique_groups = sorted(df_all["group_id"].unique().tolist())

def get_samples_by_group(g):
    df_group = df_all[df_all["group_id"] == g].copy()
    chosen_idx = sorted(df_group["sample_idx"].unique().tolist())
    return chosen_idx

# === 3) 建立 Dash 應用，配置基礎介面 ===
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H3("Group 預測 vs. 真實值 (Plotly 折線圖)"),

    # 下拉選單：選取 group
    html.Label("選擇 group："),
    dcc.Dropdown(
        id="group-dropdown",
        options=[{"label": str(g), "value": g} for g in unique_groups],
        value=unique_groups[0],
        clearable=False
    ),

    # 下拉選單：選取 sample_idx
    html.Label("選擇 sample_idx："),
    dcc.Dropdown(
        id="sample-dropdown",
        options=[],
        value=None,
        clearable=False
    ),

    # 顯示圖表
    dcc.Graph(id="prediction-graph")
])

# === 4) Callback：根據選到的 group，更新 sample-dropdown 的可選範圍 ===
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

# === 5) Callback：根據 group & sample_idx 繪圖 ===
@app.callback(
    Output("prediction-graph", "figure"),
    Input("group-dropdown", "value"),
    Input("sample-dropdown", "value")
)
def update_graph(selected_group, selected_sample):
    if (selected_group is None) or (selected_sample is None):
        return go.Figure()

    # 取出該筆資料(只會有一 row)
    df_sel = df_all[(df_all["group_id"] == selected_group) &
                    (df_all["sample_idx"] == selected_sample)]
    if df_sel.empty:
        return go.Figure()

    # 讀取 valid_length
    valid_length = int(df_sel["valid_length"].values[0])

    # 讀取 y_1..y_n, p_1..p_n
    true_values = df_sel[y_cols].values[0]
    pred_values = df_sel[p_cols].values[0]

    # 若想只畫前 valid_length 步, 做 slicing
    # e.g. true_values[:valid_length], pred_values[:valid_length]
    # 但若 valid_length=0, 代表完全沒有效, 就不要畫
    if valid_length > 0:
        true_values = true_values[:valid_length]
        pred_values = pred_values[:valid_length]
    else:
        # valid_length=0 -> 什麼都不畫
        return go.Figure()

    x_axis = list(range(1, valid_length + 1))

    # 建立 Plotly traces
    trace_true = go.Scatter(
        x=x_axis,
        y=true_values,
        mode="lines+markers",
        name="真實值"
    )
    trace_pred = go.Scatter(
        x=x_axis,
        y=pred_values,
        mode="lines+markers",
        name="預測值"
    )

    fig = go.Figure(data=[trace_true, trace_pred])
    fig.update_layout(
        title=f"Group {selected_group} / Sample {selected_sample} (valid_length={valid_length})",
        xaxis_title="預測步數 (step)",
        yaxis_title="數值"
    )
    return fig

def run_visual_tool(config):
    """
    將執行 Dash App 的邏輯包裝成一個函式，供外部程式呼叫。
    """
    # 假設 config 中若有其他設定，可在此處理
    # 例如：使用 config 中的 DB 連線路徑

    app.run_server(debug=True)

if __name__ == "__main__":
    run_visual_tool({})

