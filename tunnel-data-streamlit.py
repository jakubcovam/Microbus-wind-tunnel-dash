# Run it in terminal as:  streamlit run tunnel-data-streamlit.py
# Stop it in terminal as: Ctrl + C

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

# Dataset layout assumed:
# data/
#   base/
#     concentration/<Direction>/*.dat|*.txt
#     velocity/<Direction>/*.dat|*.txt
#   trees/
#     concentration/West/*.dat|*.txt
#     (velocity may be missing)
SCENARIOS = {
    "Bez stromů": "notrees",
    "Se stromy": "trees",
}

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
def build_file_map(direction: str, scenario_key: str) -> dict[str, dict[str, str]]:
    """
    Returns:
      { "y=..mm": { "concentration": "/path/to/file", "velocity": "/path/to/file" } }

    Note: some y-keys may have only concentration (e.g. trees scenario).
    """
    mapping: dict[str, dict[str, str]] = {}

    for kind in ["concentration", "velocity"]:
        d = os.path.join(BASE_DIR, scenario_key, kind, direction)
        if not os.path.isdir(d):
            continue

        for f in os.listdir(d):
            if not f.lower().endswith((".dat", ".txt")):
                continue
            ykey = extract_y_key(f)
            if ykey:
                mapping.setdefault(ykey, {})[kind] = os.path.join(d, f)

    return mapping

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
        # everything currently RdBu_r
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
def build_zip_bytes(direction: str, scenario_key: str) -> bytes:
    fmap = build_file_map(direction, scenario_key)
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

st.markdown(
    """
    Text text text ... (popis experimentu a dat).
    """
)

st.info(
    "Zobrazovaná pole jsou prostorové průměry na základě dostupných tunelových měření. "
    "Aplikace neprovádí interpolaci mimo rozsah dat."
)

st.info(
    "Pro stažení vybraných dat ve formátu ZIP použijte tlačítko v dolní části obrazovky."
)

st.divider()

st.markdown("**Vyberte směr větru, scénář, měřicí pozici (*y = ... mm*) a zobrazovanou veličinu:**")

# ---- controls row (direction + scenario)
direction = st.radio(
    "Směr větru",
    options=list(DIR_LABELS.keys()),
    format_func=lambda k: DIR_LABELS.get(k, k),
    horizontal=True
)

# Scenario selection:
# If trees exist only for West, force "Bez stromů" for East.
if direction != "West":
    scenario_label = "Bez stromů"
    scenario_key = SCENARIOS[scenario_label]
    st.warning("Pro tento směr je dostupný pouze scénář bez stromů.")
else:
    scenario_label = st.radio(
        "Scénář",
        options=list(SCENARIOS.keys()),
        horizontal=True
    )
    scenario_key = SCENARIOS[scenario_label]

# ---- file map for chosen direction+scenario
fmap = build_file_map(direction, scenario_key)
keys = sorted(fmap.keys())

if not keys:
    st.error(
        f"Nebyly nalezeny soubory pro: směr={direction}, scénář={scenario_label}. "
        f"Čekám data v {BASE_DIR}/{scenario_key}/..."
    )
    st.stop()

pos_key = st.selectbox("Měřicí pozice", options=keys)

# ---- variable selectors
colA, colB = st.columns(2, gap="large")

with colA:
    st.subheader("Koncentrace")
    conc_var = st.selectbox(
        "Proměnná (koncentrace)",
        options=["C", "C_std"],
        format_func=lambda v: {"C": "C*", "C_std": "C* std"}.get(v, v),
        key="conc_var"
    )

with colB:
    st.subheader("Rychlost")
    vel_var = st.selectbox(
        "Proměnná (rychlost)",
        options=["U_Uref", "W_Uref", "TKE_Uref2"],
        format_func=lambda v: {"U_Uref": "U/Uref", "W_Uref": "W/Uref", "TKE_Uref2": "TKE/Uref²"}.get(v, v),
        key="vel_var"
    )

# ---- load and plot (conditional for velocity)
paths = fmap[pos_key]
has_conc = "concentration" in paths
has_vel = "velocity" in paths

cz_dir = DIR_LABELS.get(direction, direction)

fig_c = go.Figure()
fig_v = None

if has_conc:
    dfc = load_tecplot(paths["concentration"], CONC_COLS)
    fig_c = make_field_plot(
        dfc,
        conc_var,
        f"{cz_dir} / {scenario_label} - koncentrace ({pos_key})"
    )
else:
    st.error("Pro vybranou pozici chybí soubor koncentrace.")

if has_vel:
    dfv = load_tecplot(paths["velocity"], VEL_COLS)
    fig_v = make_field_plot(
        dfv,
        vel_var,
        f"{cz_dir} / {scenario_label} - rychlost ({pos_key})"
    )

# ---- plots
col1, col2 = st.columns(2, gap="large")
with col1:
    st.plotly_chart(fig_c, width="stretch")

with col2:
    if fig_v is None:
        st.warning("Pro tento scénář nejsou k dispozici data rychlosti.")
    else:
        st.plotly_chart(fig_v, width="stretch")

st.divider()

# ---- download section
st.success("Data ke stažení zde: ⬇️")

zip_label = f"Stáhnout data - {cz_dir} / {scenario_label} (ZIP)"
zip_bytes = build_zip_bytes(direction, scenario_key)

st.download_button(
    label=zip_label,
    data=zip_bytes,
    file_name=f"{direction}_{scenario_key}_all_data.zip",
    mime="application/zip"
)
