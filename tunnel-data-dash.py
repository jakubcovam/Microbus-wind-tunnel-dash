import os
import re
import io
import zipfile
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = "data"
FIG_HEIGHT = 500

# =====================================================
# COLUMN DEFINITIONS
# =====================================================
CONC_COLS = ["x", "y", "z", "x_B", "y_B", "z_B", "C", "C_std"]

VEL_COLS = [
    "x", "y", "z",
    "x_H", "y_H", "z_H",
    "U", "W", "Length",
    "U_Uref", "W_Uref",
    "Std_U_Uref", "Std_W_Uref",
    "TKE_Uref2", "uw_Uref2",
    "Corr", "Skew_U", "Skew_W",
    "Kurt_U", "Kurt_W", "Length_Uref"
]

DIR_LABELS = {
    "East": "Východní vítr",
    "West": "Západní vítr",
}

# =====================================================
# TEC PLOT PARSER
# =====================================================
def load_tecplot(path: str, columns: list[str]) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_start = next(
        i for i, line in enumerate(lines)
        if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == "-")
    )

    df = pd.read_csv(path, sep=r"\s+", skiprows=data_start, header=None)
    df.columns = columns

    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x", "y", "z"])

# =====================================================
# FILE PAIRING BY y=XXmm
# =====================================================
Y_PATTERN = re.compile(r"y\s*=\s*([-+]?\d+\.?\d*)\s*mm", re.IGNORECASE)

def extract_y_key(fname: str) -> str | None:
    m = Y_PATTERN.search(fname)
    return f"y={m.group(1)}mm" if m else None

def build_file_map(direction: str):
    mapping = {}
    for kind in ["concentration", "velocity"]:
        d = os.path.join(BASE_DIR, kind, direction)
        for f in os.listdir(d):
            if not f.lower().endswith((".dat", ".txt")):
                continue
            ykey = extract_y_key(f)
            if ykey:
                mapping.setdefault(ykey, {})[kind] = os.path.join(d, f)
    return {k: v for k, v in mapping.items()
            if "concentration" in v and "velocity" in v}

# =====================================================
# AXIS LOCK
# =====================================================
def lock_axes(fig, xvals, yvals):
    fig.update_layout(
        xaxis=dict(range=[float(xvals.min()), float(xvals.max())],
                   autorange=False, fixedrange=True),
        yaxis=dict(range=[float(yvals.min()), float(yvals.max())],
                   autorange=False, fixedrange=True)
    )

# =====================================================
# FIELD PLOTTER
# =====================================================
def make_field_plot(df: pd.DataFrame, variable: str, title: str) -> go.Figure:
    def choose_colormap(variable):
        if variable in ["W_Uref", "uw_Uref2", "Corr"]:
            return "RdBu_r"
        if variable in ["U_Uref"]:
            return "RdBu_r"  #"Cividis"
        if variable in ["TKE_Uref2"]:
            return "RdBu_r" #"Magma"
        return "RdBu_r" #"Viridis"

    fig = go.Figure()
    nx, ny, nz = df["x"].nunique(), df["y"].nunique(), df["z"].nunique()

    # ---------------- x–y ----------------
    if nx > 1 and ny > 1:
        grid = df.pivot_table(index="y", columns="x", values=variable, aggfunc="mean")
        xv, yv = grid.columns.to_numpy(float), grid.index.to_numpy(float)

        if abs(grid.values.max() - grid.values.min()) < 1e-12:
            fig.add_trace(go.Heatmap(
                x=xv, y=yv, z=grid.values,
                colorscale=choose_colormap(variable),
                colorbar=dict(title=variable)
            ))
        else:
            fig.add_trace(go.Contour(
                x=xv, y=yv, z=grid.values,
                colorscale=choose_colormap(variable),
                contours=dict(showlines=False),
                colorbar=dict(title=variable)
            ))

        lock_axes(fig, xv, yv)
        fig.update_xaxes(title="x [mm]")
        fig.update_yaxes(title="y [mm]", scaleanchor="x")

    # ---------------- x–z ----------------
    elif nx > 1 and nz > 1:
        zcol, zlabel = ("z", "z [mm]")
        if df["z"].max() > 1000:
            zcol, zlabel = ("z_H", "z/H")

        grid = df.pivot_table(index=zcol, columns="x", values=variable, aggfunc="mean")
        xv, zv = grid.columns.to_numpy(float), grid.index.to_numpy(float)

        if abs(grid.values.max() - grid.values.min()) < 1e-12:
            fig.add_trace(go.Heatmap(
                x=xv, y=zv, z=grid.values,
                colorscale=choose_colormap(variable),
                colorbar=dict(title=variable)
            ))
        else:
            fig.add_trace(go.Contour(
                x=xv, y=zv, z=grid.values,
                colorscale=choose_colormap(variable),
                contours=dict(showlines=False),
                colorbar=dict(title=variable)
            ))

        lock_axes(fig, xv, zv)
        fig.update_xaxes(title="x [mm]")
        fig.update_yaxes(title=zlabel)

    # ---------------- fallback ----------------
    else:
        fig.add_trace(go.Scatter(
            x=df["x"], y=df[variable], mode="lines+markers"
        ))
        fig.update_xaxes(title="x [mm]")
        fig.update_yaxes(title=variable)

    fig.update_layout(
        title=title,
        height=FIG_HEIGHT,
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=50),
        autosize=False
    )

    return fig

# =====================================================
# DASH APP
# =====================================================
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"width": "1200px", "margin": "auto"},
    children=[
        html.H2("Tunelová měření - pole průměrných koncentrací a rychlostí"),

        html.Label("Směr větru"),
        dcc.RadioItems(
            id="dir-radio",
            options=[
                {"label": label, "value": key}
                for key, label in DIR_LABELS.items()
            ],
            value="East",
            inline=True
        ),

        html.Br(),

        html.Label("Měřicí pozice"),
        dcc.Dropdown(id="pos-dropdown", clearable=False),

        html.Br(),

        html.Div(
            style={"display": "flex", "gap": "20px"},
            children=[
                html.Div(
                    style={"width": "50%"},
                    children=[
                        html.H4("Koncentrace"),
                        dcc.Dropdown(
                            id="conc-var",
                            options=[
                                {"label": "C*", "value": "C"},
                                {"label": "C* std", "value": "C_std"},
                            ],
                            value="C",
                            clearable=False
                        ),
                        dcc.Graph(
                            id="conc-plot",
                            style={"height": f"{FIG_HEIGHT}px"},
                            config={"responsive": True}
                        )
                    ]
                ),
                html.Div(
                    style={"width": "50%"},
                    children=[
                        html.H4("Rychlost"),
                        dcc.Dropdown(
                            id="vel-var",
                            options=[
                                {"label": "U/Uref", "value": "U_Uref"},
                                {"label": "W/Uref", "value": "W_Uref"},
                                {"label": "TKE/Uref²", "value": "TKE_Uref2"},
                            ],
                            value="U_Uref",
                            clearable=False
                        ),
                        dcc.Graph(
                            id="vel-plot",
                            style={"height": f"{FIG_HEIGHT}px"},
                            config={"responsive": True}
                        )
                    ]
                )
            ]
        ),

        html.Br(),

        html.Button(id="download-all-btn"),
        dcc.Download(id="download-all"),

    ]
)

# =====================================================
# CALLBACKS
# =====================================================
@app.callback(
    Output("pos-dropdown", "options"),
    Output("pos-dropdown", "value"),
    Input("dir-radio", "value")
)
def update_positions(direction):
    fmap = build_file_map(direction)
    keys = sorted(fmap.keys())
    return ([{"label": k, "value": k} for k in keys],
            keys[0] if keys else None)

@app.callback(
    Output("conc-plot", "figure"),
    Output("vel-plot", "figure"),
    Input("dir-radio", "value"),
    Input("pos-dropdown", "value"),
    Input("conc-var", "value"),
    Input("vel-var", "value")
)
def update_plots(direction, pos_key, conc_var, vel_var):
    if pos_key is None:
        return go.Figure(), go.Figure()

    fmap = build_file_map(direction)
    dfc = load_tecplot(fmap[pos_key]["concentration"], CONC_COLS)
    dfv = load_tecplot(fmap[pos_key]["velocity"], VEL_COLS)

    cz_dir = DIR_LABELS.get(direction, direction)

    return (
        make_field_plot(
            dfc,
            conc_var,
            f"{cz_dir} – koncentrace ({pos_key})"
        ),
        make_field_plot(
            dfv,
            vel_var,
            f"{cz_dir} – rychlost ({pos_key})"
        )
    )

@app.callback(
    Output("download-all-btn", "children"),
    Input("dir-radio", "value")
)
def update_download_button_label(direction):
    cz_dir = DIR_LABELS.get(direction, direction)
    return f"Stáhnout data – {cz_dir} (ZIP)"

@app.callback(
    Output("download-all", "data"),
    Input("download-all-btn", "n_clicks"),
    State("dir-radio", "value"),
    prevent_initial_call=True
)

def download_all_data(n_clicks, direction):
    fmap = build_file_map(direction)
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for ykey, paths in fmap.items():
            for kind, path in paths.items():
                arcname = f"{kind}/{os.path.basename(path)}"
                zf.write(path, arcname=arcname)

    buffer.seek(0)
    return dcc.send_bytes(buffer.read(),
                          filename=f"{direction}_all_data.zip")

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)