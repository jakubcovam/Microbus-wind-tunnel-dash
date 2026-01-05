# Run it in terminal as: > streamlit run tunnel-data-streamlit.py
# Stop it in terminal as: > Ctrl + C

import os
import re
import io
import zipfile
import pandas as pd
import streamlit as st
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
@st.cache_data(show_spinner=False)
def load_tecplot(path: str, columns: list[str]) -> pd.DataFrame:
    # NOTE: caching uses file path as key; if files change often, consider adding mtime to the cache key.
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

@st.cache_data(show_spinner=False)
def build_file_map(direction: str):
    mapping = {}
    for kind in ["concentration", "velocity"]:
        d = os.path.join(BASE_DIR, kind, direction)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.lower().endswith((".dat", ".txt")):
                continue
            ykey = extract_y_key(f)
            if ykey:
                mapping.setdefault(ykey, {})[kind] = os.path.join(d, f)

    # keep only complete pairs
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
            return "RdBu_r"
        if variable in ["TKE_Uref2"]:
            return "RdBu_r"
        return "RdBu_r"

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
# ZIP BUILDER
# =====================================================
def build_zip_bytes(direction: str) -> bytes:
    fmap = build_file_map(direction)
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for ykey, paths in fmap.items():
            for kind, path in paths.items():
                arcname = f"{kind}/{os.path.basename(path)}"
                zf.write(path, arcname=arcname)

    buffer.seek(0)
    return buffer.read()

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Tunelová měření", layout="wide")

st.title("Tunelová měření - pole průměrných koncentrací a rychlostí")

# controls
direction = st.radio(
    "Směr větru",
    options=list(DIR_LABELS.keys()),
    format_func=lambda k: DIR_LABELS.get(k, k),
    horizontal=True
)

fmap = build_file_map(direction)
keys = sorted(fmap.keys())

if not keys:
    st.error(f"Nebyly nalezeny páry souborů pro směr: {direction} (čekám data v {BASE_DIR}/...)")
    st.stop()

pos_key = st.selectbox("Měřicí pozice", options=keys)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Koncentrace")
    conc_var = st.selectbox(
        "Proměnná (koncentrace)",
        options=["C", "C_std"],
        format_func=lambda v: {"C": "C*", "C_std": "C* std"}.get(v, v),
        key="conc_var"
    )

with col2:
    st.subheader("Rychlost")
    vel_var = st.selectbox(
        "Proměnná (rychlost)",
        options=["U_Uref", "W_Uref", "TKE_Uref2"],
        format_func=lambda v: {"U_Uref": "U/Uref", "W_Uref": "W/Uref", "TKE_Uref2": "TKE/Uref²"}.get(v, v),
        key="vel_var"
    )

# load + plot
dfc = load_tecplot(fmap[pos_key]["concentration"], CONC_COLS)
dfv = load_tecplot(fmap[pos_key]["velocity"], VEL_COLS)

cz_dir = DIR_LABELS.get(direction, direction)

fig_c = make_field_plot(dfc, conc_var, f"{cz_dir} - koncentrace ({pos_key})")
fig_v = make_field_plot(dfv, vel_var, f"{cz_dir} - rychlost ({pos_key})")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.plotly_chart(fig_c, use_container_width=True)
with col2:
    st.plotly_chart(fig_v, use_container_width=True)

st.divider()

zip_label = f"Stáhnout data - {cz_dir} (ZIP)"
zip_bytes = build_zip_bytes(direction)

st.download_button(
    label=zip_label,
    data=zip_bytes,
    file_name=f"{direction}_all_data.zip",
    mime="application/zip"
)
