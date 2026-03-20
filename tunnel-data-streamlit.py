# Run it in terminal as:  streamlit run tunnel-data-streamlit.py
# Stop it in terminal as: Ctrl + C

import os
import re
import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# =====================================================
# CONFIG
# =====================================================
BASE_DIR = "data"
FIG_HEIGHT = 500

SCENARIOS = {
    "Bez stromů": "notrees",
    "Se stromy":  "trees",
}

DIR_LABELS = {
    "East": "Východní vítr",
    "West": "Západní vítr",
}

PLANE_LABELS = {
    "vertical":   "Vertikální roviny (x–z)",
    "horizontal": "Horizontální rovina (x–y)",
}

# =====================================================
# COLUMN DEFINITIONS
# =====================================================
CONC_COLS_FFID = ["x", "y", "z", "x_B", "y_B", "z_B", "C", "C_std"]
CONC_COLS_PIV  = ["x", "y", "z", "C"]

VEL_COLS_VERTICAL = [
    "x", "y", "z", "x_H", "y_H", "z_H",
    "U", "W", "Length",
    "U_Uref", "W_Uref",
    "Std_U_Uref", "Std_W_Uref",
    "TKE_Uref2", "uw_Uref2",
    "Corr", "Skew_U", "Skew_W", "Kurt_U", "Kurt_W", "Length_Uref",
]

# Horizontal plane: V component instead of W
VEL_COLS_HORIZONTAL = [
    "x", "y", "z", "x_H", "y_H", "z_H",
    "U", "V", "Length",
    "U_Uref", "V_Uref",
    "Std_U_Uref", "Std_V_Uref",
    "TKE_Uref2", "uv_Uref2",
    "Corr", "Skew_U", "Skew_W", "Kurt_U", "Kurt_W", "Length_Uref",
]

# =====================================================
# COLORSCALE & COLORBAR DEFINITIONS
# =====================================================
VARIABLE_DISPLAY = {
    "C":         ("Viridis", "C* [–]",          False),
    "C_std":     ("Viridis", "C* std [–]",       False),
    "U_Uref":    ("RdBu_r",  "U / U_ref [–]",    True),
    "W_Uref":    ("RdBu_r",  "W / U_ref [–]",    True),
    "V_Uref":    ("RdBu_r",  "V / U_ref [–]",    True),
    "TKE_Uref2": ("Plasma",  "TKE / U_ref² [–]", False),
}

def get_variable_style(variable: str) -> tuple[str, str, bool]:
    return VARIABLE_DISPLAY.get(variable, ("Viridis", variable, False))

def make_colorbar_kwargs(label: str) -> dict:
    return dict(
        title=dict(text=label, side="right", font=dict(size=13)),
        tickfont=dict(size=11),
        thickness=16,
        len=0.9,
    )

# =====================================================
# PARSERS
# =====================================================
@st.cache_data(show_spinner=False)
def load_tecplot_point(path: str, columns: list[str]) -> pd.DataFrame:
    """Load Tecplot POINT format (FFID concentration + velocity files)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_start = next(
        i for i, line in enumerate(lines)
        if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == "-")
    )
    df = pd.read_csv(path, sep=r"\s+", skiprows=data_start, header=None,
                     engine="python")
    df = df.iloc[:, :len(columns)]
    df.columns = columns
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x", "y", "z"])


@st.cache_data(show_spinner=False)
def load_tecplot_block(path: str, columns: list[str]) -> pd.DataFrame:
    """
    Load Tecplot BLOCK format (PIV concentration files).
    All numeric values are read sequentially; the first npoints*ncols
    values are then reshaped column-by-column as Tecplot BLOCK specifies.
    """
    tokens: list[float] = []
    n_points: int | None = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if n_points is None:
                m = re.search(r"I\s*=\s*(\d+).*?J\s*=\s*(\d+)", line, re.IGNORECASE)
                if m:
                    n_points = int(m.group(1)) * int(m.group(2))
            for tok in line.split():
                try:
                    tokens.append(float(tok))
                except ValueError:
                    pass
            if n_points and len(tokens) >= n_points * len(columns):
                break

    if n_points is None or len(tokens) < n_points * len(columns):
        raise ValueError(f"Nelze načíst BLOCK soubor: {path}")

    arr = np.array(tokens[: n_points * len(columns)]).reshape(
        len(columns), n_points
    ).T
    df = pd.DataFrame(arr, columns=columns)
    return df.dropna(subset=["x", "y", "z"])

# =====================================================
# FILE MAP BUILDER
# =====================================================
Y_PATTERN = re.compile(r"y\s*=\s*([-+]?\d+\.?\d*)\s*mm", re.IGNORECASE)
Z_PATTERN = re.compile(r"z\s*=\s*([-+]?\d+\.?\d*)\s*mm", re.IGNORECASE)

def _extract_key(fname: str, pattern: re.Pattern, prefix: str) -> str | None:
    m = pattern.search(fname)
    return f"{prefix}={m.group(1)}mm" if m else None


@st.cache_data(show_spinner=False)
def build_file_map(direction: str, scenario_key: str, plane: str) -> dict[str, dict]:
    """
    Returns dict keyed by position label with sub-keys:
      'ffid'       → path  (FFID concentration)
      'piv'        → path  (PIV concentration, vertical only)
      'velocity'   → path
      'vel_format' → 'vertical' | 'horizontal'
    """
    mapping: dict[str, dict] = {}

    if plane == "vertical":
        # FFID concentration
        d = os.path.join(BASE_DIR, scenario_key, "concentration", direction)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if not f.lower().endswith((".dat", ".txt")):
                    continue
                key = _extract_key(f, Y_PATTERN, "y")
                if key:
                    mapping.setdefault(key, {})["ffid"] = os.path.join(d, f)

        # PIV concentration
        piv_dir = os.path.join(BASE_DIR, scenario_key, "concentration", direction, "PIV")
        if os.path.isdir(piv_dir):
            for f in os.listdir(piv_dir):
                if not f.lower().endswith((".dat", ".txt")):
                    continue
                key = _extract_key(f, Y_PATTERN, "y")
                if key:
                    mapping.setdefault(key, {})["piv"] = os.path.join(piv_dir, f)

        # Velocity
        vel_dir = os.path.join(BASE_DIR, scenario_key, "velocity", direction)
        if os.path.isdir(vel_dir):
            for f in os.listdir(vel_dir):
                if not f.lower().endswith((".dat", ".txt")):
                    continue
                key = _extract_key(f, Y_PATTERN, "y")
                if key:
                    mapping.setdefault(key, {})["velocity"]   = os.path.join(vel_dir, f)
                    mapping[key]["vel_format"] = "vertical"

    else:  # horizontal
        # FFID concentration
        horiz_conc = os.path.join(BASE_DIR, scenario_key, "concentration", direction, "horizontal")
        if os.path.isdir(horiz_conc):
            for f in os.listdir(horiz_conc):
                if not f.lower().endswith((".dat", ".txt")):
                    continue
                key = _extract_key(f, Z_PATTERN, "z")
                if key:
                    mapping.setdefault(key, {})["ffid"] = os.path.join(horiz_conc, f)

        # Velocity
        horiz_vel = os.path.join(BASE_DIR, scenario_key, "velocity", direction, "horizontal")
        if os.path.isdir(horiz_vel):
            for f in os.listdir(horiz_vel):
                if not f.lower().endswith((".dat", ".txt")):
                    continue
                key = _extract_key(f, Z_PATTERN, "z") or os.path.splitext(f)[0]
                mapping.setdefault(key, {})["velocity"]   = os.path.join(horiz_vel, f)
                mapping[key]["vel_format"] = "horizontal"

    return mapping

# =====================================================
# AXIS LOCK & ASPECT RATIO
# =====================================================
def lock_axes(fig, xvals, yvals):
    fig.update_layout(
        xaxis=dict(range=[float(xvals.min()), float(xvals.max())],
                   autorange=False, fixedrange=True),
        yaxis=dict(range=[float(yvals.min()), float(yvals.max())],
                   autorange=False, fixedrange=True),
    )

def compute_fig_height(xvals, yvals, base_width_px=700, min_h=280, max_h=700) -> int:
    x_range = float(xvals.max() - xvals.min())
    y_range = float(yvals.max() - yvals.min())
    if x_range < 1e-9:
        return FIG_HEIGHT
    h = int(base_width_px * y_range / x_range) + 100
    return max(min_h, min(h, max_h))

# =====================================================
# FIELD PLOTTER
# =====================================================
def make_field_plot(df: pd.DataFrame, variable: str, title: str,
                    plane: str = "vertical") -> go.Figure:
    colorscale, colorbar_label, is_diverging = get_variable_style(variable)
    colorbar_kw = make_colorbar_kwargs(colorbar_label)

    fig = go.Figure()
    nx, ny, nz = df["x"].nunique(), df["y"].nunique(), df["z"].nunique()
    fig_height = FIG_HEIGHT

    def _add_trace(xv, yv, zmat, ylabel):
        nonlocal fig_height
        zmin_v, zmax_v = float(np.nanmin(zmat)), float(np.nanmax(zmat))
        cscale_kw = {}
        if is_diverging:
            abs_max = max(abs(zmin_v), abs(zmax_v))
            cscale_kw = dict(zmin=-abs_max, zmax=abs_max)
        flat = abs(zmax_v - zmin_v) < 1e-12
        TraceClass = go.Heatmap if flat else go.Contour
        extra = {} if flat else dict(contours=dict(showlines=False))
        fig.add_trace(TraceClass(
            x=xv, y=yv, z=zmat,
            colorscale=colorscale,
            colorbar=colorbar_kw,
            **cscale_kw, **extra,
        ))
        lock_axes(fig, np.array(xv), np.array(yv))
        fig.update_xaxes(title_text="x [mm]", title_font=dict(size=13))
        fig.update_yaxes(title_text=ylabel,    title_font=dict(size=13))
        fig_height = compute_fig_height(np.array(xv), np.array(yv))

    # x–y (horizontal plane or single z layer)
    if nx > 1 and ny > 1 and (plane == "horizontal" or nz == 1):
        grid = df.pivot_table(index="y", columns="x", values=variable, aggfunc="mean")
        xv, yv = grid.columns.to_numpy(float), grid.index.to_numpy(float)
        _add_trace(xv, yv, grid.values, "y [mm]")
        fig.update_yaxes(scaleanchor="x")

    # x–z (vertical plane)
    elif nx > 1 and nz > 1:
        use_norm = df["z"].max() > 1000
        zcol   = "z_H" if use_norm else "z"
        zlabel = "z / H [–]" if use_norm else "z [mm]"
        grid = df.pivot_table(index=zcol, columns="x", values=variable, aggfunc="mean")
        xv, zv = grid.columns.to_numpy(float), grid.index.to_numpy(float)
        _add_trace(xv, zv, grid.values, zlabel)

    # fallback
    else:
        fig.add_trace(go.Scatter(x=df["x"], y=df[variable], mode="lines+markers"))
        fig.update_xaxes(title_text="x [mm]",      title_font=dict(size=13))
        fig.update_yaxes(title_text=colorbar_label, title_font=dict(size=13))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=fig_height,
        template="plotly_white",
        margin=dict(l=70, r=20, t=55, b=60),
        autosize=True,
    )
    return fig

# =====================================================
# ZIP BUILDER
# =====================================================
def build_zip_bytes(direction: str, scenario_key: str, plane: str) -> bytes:
    fmap = build_file_map(direction, scenario_key, plane)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for pos_key, paths in fmap.items():
            for kind, path in paths.items():
                if kind == "vel_format":
                    continue
                arcname = f"{kind}/{os.path.basename(path)}"
                zf.write(path, arcname=arcname)
    buffer.seek(0)
    return buffer.read()

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Tunelová měření", layout="wide")

st.title("Tunelová měření – pole průměrných koncentrací a rychlostí")

col_text, col_img = st.columns([3, 2], gap="large")

with col_text:
    st.markdown(
        """
        Databáze obsahuje výsledky měření z aerodynamického tunelu v Novém Kníně 
        [Ústavu termomechaniky AV ČR, v. v. i.](https://www.it.cas.cz/d1/l013/)
        Jedná se o výstupy měření na fyzikálním modelu **Legerovy ulice v Praze** v měřítku **1 : 500**.
        Měření byla provedena v rámci projektu [Microbus](https://www.microbus.cz/) financováném 
        Technologickou agenturou ČR v rámci programu Prostředí pro život 2 v letech 2025–2027.

        **Geometrie a souřadnicový systém**

        Souřadnicový systém (x, y, z) je vztažen k modelu ulice. Vertikální roviny (x–z) jsou kolmé na osu zdroje (y),
        horizontální rovina (x–y) byla měřena ve výšce z = 7,5 mm nebo z = 8 mm. Všechny rozměry jsou v mm v tunelovém měřítku.
        Bezrozměrné souřadnice x/B, y/B, x/H, y/H jsou rovněž součástí datových souborů,
        kde B = H je charakteristická šířka ulice, resp. výška budov.

        **Měřicí techniky**

        - **PIV** (Particle Image Velocimetry) – plošné časově rozlišené měření 2D vektorů rychlostí a koncentrací
          pasivního polutantu (aerosolové částice < 2,5 µm)
        - **FFID** (Fast Flame Ionization Detector) – bodové měření koncentrací etanu

        **Normování veličin**

        Složky rychlosti a jejich statistické momenty jsou normovány referenční rychlostí volného proudu
        U_ref = 6 m/s. Koncentrace polutantu jsou prezentovány v bezrozměrné formě C* = C·B·U_ref·L / Q,
        kde C je změřená koncentrace etanu [ppm], B = 0,046 m je průměrná šířka kaňonu, L = 1 m je délka zdroje
        a Q = 18 ml/s je objemový průtok etanu za standardních podmínek.

        **Formát dat**

        Soubory jsou v ASCII formátu (přípona `.dat`, formát Tecplot).
        Hlavička `VARIABLES` v každém souboru uvádí názvy sloupců. Data jsou členěna podle směru větru
        (`West`, `East`) a typu veličiny (`concentration`, `velocity`).
        Název souboru kóduje pozici měřené roviny (např. `y = -115 mm`).
        """
    )

with col_img:
    st.image("img/planes.png", width='stretch')

st.info(
    "Zobrazovaná pole jsou prostorové průměry na základě dostupných tunelových měření. "
    "Aplikace neprovádí interpolaci mimo rozsah dat."
)
st.info("Pro stažení dat zvoleného nastavení ve formátu ZIP použijte tlačítko v levé části obrazovky.")

st.divider()

# =====================================================
# SIDEBAR – controls
# =====================================================
with st.sidebar:
    st.header("Nastavení zobrazení")

    # ---- direction ----
    direction = st.radio(
        "Směr větru",
        options=list(DIR_LABELS.keys()),
        format_func=lambda k: DIR_LABELS[k],
    )

    # ---- scenario ----
    if direction != "West":
        scenario_label = "Bez stromů"
        scenario_key   = SCENARIOS[scenario_label]
        st.warning("Pro tento směr je dostupný pouze scénář bez stromů.")
    else:
        scenario_label = st.radio(
            "Scénář",
            options=list(SCENARIOS.keys()),
        )
        scenario_key = SCENARIOS[scenario_label]

    # ---- plane type ----
    available_planes = ["vertical", "horizontal"] if scenario_key == "notrees" else ["vertical"]
    plane = st.radio(
        "Typ roviny",
        options=available_planes,
        format_func=lambda p: PLANE_LABELS[p],
    )

    # ---- position ----
    fmap = build_file_map(direction, scenario_key, plane)
    keys = sorted(fmap.keys())

    if not keys:
        st.error(
            f"Nebyly nalezeny soubory pro: směr={direction}, scénář={scenario_label}, "
            f"rovina={PLANE_LABELS[plane]}."
        )
        st.stop()

    pos_key = st.selectbox("Měřicí pozice", options=keys)
    paths   = fmap[pos_key]

    has_ffid   = "ffid"     in paths
    has_piv    = "piv"      in paths
    has_vel    = "velocity" in paths
    vel_format = paths.get("vel_format", "vertical")

    st.divider()

    # ---- concentration source ----
    if plane == "vertical" and has_ffid and has_piv:
        conc_source = st.radio(
            "Zdroj dat koncentrace",
            options=["FFID", "PIV"],
            help="FFID = bodové měření etanu · PIV = plošná intenzita částic (kalibrovaná)",
        )
    elif has_piv and not has_ffid:
        conc_source = "PIV"
    else:
        conc_source = "FFID"

    # ---- concentration variable ----
    conc_var_options = ["C", "C_std"] if (conc_source == "FFID" and has_ffid) else ["C"]
    conc_var_labels  = {"C": "C* [–]", "C_std": "C* std [–]"}
    conc_var = st.selectbox(
        "Proměnná – koncentrace",
        options=conc_var_options,
        format_func=lambda v: conc_var_labels.get(v, v),
        key="conc_var",
    )

    # ---- velocity variable ----
    if plane == "horizontal":
        vel_var_options = ["U_Uref", "V_Uref", "TKE_Uref2"]
        vel_var_labels  = {
            "U_Uref":    "U / U_ref [–]",
            "V_Uref":    "V / U_ref [–]",
            "TKE_Uref2": "TKE / U_ref² [–]",
        }
    else:
        vel_var_options = ["U_Uref", "W_Uref", "TKE_Uref2"]
        vel_var_labels  = {
            "U_Uref":    "U / U_ref [–]",
            "W_Uref":    "W / U_ref [–]",
            "TKE_Uref2": "TKE / U_ref² [–]",
        }
    vel_var = st.selectbox(
        "Proměnná – rychlost",
        options=vel_var_options,
        format_func=lambda v: vel_var_labels.get(v, v),
        key="vel_var",
    )

    st.divider()

    # ---- download (in sidebar for easy access) ----
    zip_bytes = build_zip_bytes(direction, scenario_key, plane)
    cz_dir_sidebar = DIR_LABELS[direction]
    st.download_button(
        label="⬇️ Stáhnout data (ZIP)",
        data=zip_bytes,
        file_name=f"{direction}_{scenario_key}_{plane}_data.zip",
        mime="application/zip",
        use_container_width=True,
    )

# =====================================================
# LOAD & PLOT
# =====================================================
cz_dir = DIR_LABELS[direction]
fig_c  = go.Figure()
fig_v  = None

# --- concentration ---
if conc_source == "PIV" and has_piv:
    try:
        dfc = load_tecplot_block(paths["piv"], CONC_COLS_PIV)
        fig_c = make_field_plot(
            dfc, conc_var,
            f"{cz_dir} / {scenario_label} – PIV koncentrace ({pos_key})",
            plane=plane,
        )
    except Exception as e:
        st.error(f"Chyba při načítání PIV dat: {e}")
elif has_ffid:
    try:
        dfc = load_tecplot_point(paths["ffid"], CONC_COLS_FFID)
        fig_c = make_field_plot(
            dfc, conc_var,
            f"{cz_dir} / {scenario_label} – FFID koncentrace ({pos_key})",
            plane=plane,
        )
    except Exception as e:
        st.error(f"Chyba při načítání FFID dat: {e}")
else:
    st.warning("Pro tento scénář/pozici nejsou k dispozici data koncentrace.")

# --- velocity ---
if has_vel:
    try:
        vel_cols = VEL_COLS_HORIZONTAL if vel_format == "horizontal" else VEL_COLS_VERTICAL
        dfv = load_tecplot_point(paths["velocity"], vel_cols)
        fig_v = make_field_plot(
            dfv, vel_var,
            f"{cz_dir} / {scenario_label} – rychlost ({pos_key})",
            plane=plane,
        )
    except Exception as e:
        st.error(f"Chyba při načítání dat rychlosti: {e}")

# ---- unify heights ----
if fig_v is not None:
    unified_h = max(
        fig_c.layout.height or FIG_HEIGHT,
        fig_v.layout.height or FIG_HEIGHT,
    )
    fig_c.update_layout(height=unified_h)
    fig_v.update_layout(height=unified_h)

# ---- render ----
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("Koncentrace")
    st.plotly_chart(fig_c)
with col2:
    st.subheader("Rychlost")
    if fig_v is None:
        st.warning("Pro tento scénář/pozici nejsou k dispozici data rychlosti.")
    else:
        st.plotly_chart(fig_v)