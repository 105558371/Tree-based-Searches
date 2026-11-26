"""Three Streamlit views over the same replay services."""
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from services import data_service as ds, metrics_service as metrics, model_service as models, routing_service as routing

PAGES = ("Route Guidance", "Traffic Forecast", "Model Evaluation")
TIMES = [f"{h:02}:{m:02}" for h in range(24) for m in range(0, 60, 15)]
UNITS = "vehicles/15 min/site"
EXPECTED_ERRORS = (ValueError, KeyError, OSError, ImportError)


def time_controls(days, dataset, cfg):
    eligible = [d for d in days if dataset != ds.RESEARCH or d.year >= 2025]
    if not eligible:
        raise ds.Unavailable("No held-out dates available.")
    day = st.sidebar.selectbox("Local date", eligible, key=f"day_{dataset}")
    def preset():
        choice = st.session_state.preset
        if choice in cfg["presets"]:
            st.session_state.clock = cfg["presets"][choice]
    st.sidebar.segmented_control("Quick time", list(cfg["presets"]), key="preset", on_change=preset)
    clock = st.sidebar.select_slider("Time of day", TIMES, value=cfg["time"], key="clock")
    st.sidebar.caption("Australia/Melbourne (dataset local time) · 15-minute slots")
    return pd.Timestamp(f"{day} {clock}")


def replay_controls(dataset, cfg, days):
    when = time_controls(days, dataset, cfg)
    advanced = dataset == ds.RESEARCH and st.sidebar.checkbox("Advanced models")
    runs = models.model_choices(dataset, advanced)
    default = next((r for r in [cfg["model"], *runs] if r in runs and not models.availability(dataset, r, cfg)), runs[0])
    run = st.sidebar.selectbox("Prediction model", runs, index=runs.index(default), format_func=metrics.label,
                               key=f"model_{dataset}")
    return when, run, runs


def action(name, signature, reason, compute, *, button_area=None):
    """Only an explicit submit runs inference. Re-rendering a site/horizon uses the submitted result."""
    state = f"submitted_{name}"
    previous = st.session_state.get(state)
    if reason or (previous is not None and previous[0] != signature):
        st.session_state.pop(state, None)
    controls = st.sidebar if button_area is None else button_area
    clicked = controls.button(name, disabled=bool(reason), type="primary")
    if reason:
        st.warning(reason)
    if clicked and not reason:
        st.session_state.pop(state, None)
        try:
            with st.spinner(name + "…"):
                st.session_state[state] = signature, compute()
        except EXPECTED_ERRORS as exc:
            st.warning(str(exc))
    submitted = st.session_state.get(state)
    return submitted[1] if submitted is not None else None


def interval_text(bounds):
    a, b = bounds
    return f"{a:%Y-%m-%d %H:%M} – {b:%H:%M}"


def road_page(cfg, dataset, sites, days, build_deck):
    ids = list(sites.site_id)
    def swap():
        st.session_state.start_site, st.session_state.end_site = st.session_state.end_site, st.session_state.start_site
    for key, default in [("start_site", cfg["default_origin"]), ("end_site", cfg["default_destination"])]:
        if st.session_state.get(key) not in ids:
            st.session_state[key] = default if default in ids else ids[0]
    o = st.sidebar.selectbox("From (SCATS site)", ids, key="start_site")
    d = st.sidebar.selectbox("To (SCATS site)", ids, key="end_site")
    st.sidebar.button("Swap origin and destination", on_click=swap)
    when, model, runs = replay_controls(dataset, cfg, days)
    compare = st.sidebar.multiselect("Compare models", [r for r in runs if r != model], format_func=metrics.label)
    k = st.sidebar.slider("Number of routes", 1, 5, cfg["k_routes"])
    basemap = st.sidebar.checkbox("Load online street map", key="street_map")
    reasons = [models.availability(dataset, r, cfg) for r in [model, *compare]]
    try:
        graph = routing.road_graph(cfg, sites)
    except EXPECTED_ERRORS as exc:
        reasons.append(f"Directed road graph unavailable. {exc}")
        graph = None
    def calculate():
        return {r: routing.routes(graph, models.forecast(dataset, r, when, cfg), o, d, k) for r in [model, *compare]}
    result = action("Find Routes", (dataset, when, model, tuple(compare), o, d, k),
                    " ".join(dict.fromkeys(r for r in reasons if r)), calculate, button_area=st)
    left, center, right = st.columns([1.1, 2.7, 1.2])
    left.subheader("Routes")
    right.subheader("Summary")
    if result is None:
        center.pydeck_chart(build_deck(sites, o, d, [], basemap=basemap), height=560)
        center.caption("SCATS sites · A = origin · B = destination")
        return
    if not result[model]:
        center.info("No directed path between these sites.")
        return
    shown = left.radio("Model", list(result), format_func=metrics.label, horizontal=True) if compare else model
    routes = result[shown]
    best = routes[0]["travel_time_seconds"]
    def route_label(i):
        row = routes[i]
        delta = (row["travel_time_seconds"] - best) / 60
        suffix = "Fastest" if i == 0 else f"+{delta:.1f} min"
        return f"{row['travel_time_seconds']/60:.1f} min ({row['distance_km']:.2f} km) · {suffix}"
    picked = left.radio("Route", list(range(len(routes))), format_func=route_label)
    row = routes[picked]
    pos = sites.set_index("site_id")[["lon", "lat"]]
    lines = [dict(coords=r["geometry"] or pos.loc[list(r["site_ids"])].values.tolist(),
                  minutes=r["travel_time_seconds"]/60, km=r["distance_km"], schematic=r["geometry"] is None)
             for r in routes]
    with center:
        st.pydeck_chart(build_deck(sites, o, d, lines, picked, basemap=basemap), height=560)
        st.caption("A = origin · B = destination · Blue = selected route · Grey = alternatives. "
                   "Dashed links are schematic. Solid lines follow supplied road geometry.")
    with right:
        st.metric("Est. travel time", f"{row['travel_time_seconds']/60:.1f} min")
        st.metric("Distance", f"{row['distance_km']:.2f} km")
        st.metric("SCATS sites on route", len(row["site_ids"]))
        st.caption(f"{metrics.label(shown)} · {row['algorithm']} · {interval_text(row['target_interval'])}")
        if o == d:
            st.info("Origin equals destination. No travel required.")
        if compare:
            st.dataframe([{"Model": metrics.label(r), "Fastest ETA (min)": group[0]["travel_time_seconds"]/60}
                          for r, group in result.items() if group], hide_index=True)
    with st.expander("Route details"):
        st.write(" → ".join(map(str, row["site_ids"])))
        st.dataframe(pd.DataFrame(row["edge_details"]), hide_index=True)
        st.write(f"Total {row['travel_time_seconds']:.2f} s")
    st.caption("Each directed link uses its destination site's departure-slot flow ×4 to estimate hourly volume. "
               "Travel times use one static snapshot and include the core intersection delay.")


def forecast_map(sites, forecast, horizon, selected):
    values = forecast.snapshot(horizon)
    rows = sites.assign(flow=sites.site_id.map(values))
    rows["color"] = [[52, 168, 83] if v < 150 else [245, 166, 35] if v < 375 else [211, 74, 64] for v in rows.flow]
    rows["radius"] = np.where(rows.site_id == selected, 115, 75)
    rows["label"] = [f"SCATS {s} · {v:.1f} {UNITS}" for s, v in zip(rows.site_id, rows.flow)]
    layer = pdk.Layer("ScatterplotLayer", rows, get_position="[lon, lat]", get_fill_color="color",
                      get_radius="radius", radius_min_pixels=5, pickable=True)
    view = pdk.ViewState(latitude=sites.lat.mean(), longitude=sites.lon.mean(), zoom=11)
    return pdk.Deck(layers=[layer], initial_view_state=view, map_style=None, map_provider=None, tooltip={"text": "{label}"})


def site_chart(f, site):
    col = f.ids.index(site)
    slots = [f.interval(h)[0] for h in f.horizons]
    history = f.history[site].rename("Observed").to_frame()
    targets = pd.DataFrame({"Observed": f.actual[:, col], "Predicted": f.flows[:, col]}, index=slots)
    series = pd.concat([history, targets]).rename_axis("Interval start")
    st.line_chart(series, y_label=UNITS, x_label="Melbourne interval start", color=["#60666d", "#1a73e8"])
    st.dataframe(targets.rename_axis("Interval start"), column_config={
        "Predicted": st.column_config.NumberColumn(format="%.2f"),
        "Observed": st.column_config.NumberColumn(format="%.2f")})
    if targets.Observed.isna().all():
        st.info("No observed targets for this site in the selected window.")


def forecast_page(cfg, dataset, sites, days):
    when, model, _ = replay_controls(dataset, cfg, days)
    reason = models.availability(dataset, model, cfg)
    site = st.sidebar.selectbox("SCATS site", list(sites.site_id), key=f"site_{dataset}")
    horizons = models.horizons(dataset, model, cfg)
    default = horizons.index(cfg["horizon"]) if cfg["horizon"] in horizons else 0
    h = st.sidebar.selectbox("Horizon (minutes)", horizons, index=default if horizons else None,
                             disabled=not horizons)
    if not reason:
        st.caption(f"{metrics.label(model)} · {when:%Y-%m-%d %H:%M} Melbourne · {len(sites)} SCATS sites")
        st.write("Select Forecast to show the traffic map and site predictions for this time.")
    f = action("Forecast", (dataset, model, when), reason,
               lambda: models.forecast(dataset, model, when, cfg), button_area=st)
    if f is None:
        return
    if model in (*models.MEMBERS, models.ENSEMBLE):
        st.caption("This retrospective recipe includes archived weather and recorded AFL attendance.")
    if h not in f.horizons:
        st.warning("Selected horizon unavailable for this model.")
        return
    st.caption(f"{metrics.label(model)} · {interval_text(f.interval(h))} · {len(f.ids)} SCATS sites")
    st.pydeck_chart(forecast_map(sites, f, h, site), height=440)
    st.caption("Predicted flow in vehicles/15 min/site · Green: Low (<150) · Amber: Moderate (150–<375) · Red: High (≥375)")
    st.metric(f"SCATS {site} · {UNITS}", f"{f.snapshot(h)[site]:.1f}")
    site_chart(f, site)


def evaluation_page(cfg, dataset, sites, days):
    st.subheader("Recorded evaluation results")
    docs = metrics.records(dataset)
    if not docs:
        st.info("No verified evaluation for this dataset. Supply October 2006 model and evaluation provenance.")
        legacy = ds.GUI / "results/metrics.csv"
        if dataset == ds.ASSIGNMENT and legacy.is_file():
            with st.expander("Prototype experiment results · dataset provenance unverified"):
                st.caption("Saved GUI grid runs use a separate split and preprocessed approach averages. "
                           "The CSV does not record its source dataset.")
                st.dataframe(pd.read_csv(legacy), hide_index=True)
        return
    metric = st.sidebar.selectbox("Metric", ["MAE", "RMSE", "MAPE"])
    h = st.sidebar.selectbox("Evaluation horizon (minutes)", ds.HORIZONS, index=ds.HORIZONS.index(cfg["horizon"]))
    defaults = [r for r in ("lstm", "gru", "gwn", metrics.ENSEMBLE) if r in docs]
    runs = list(docs) if st.sidebar.checkbox("Advanced experiments") else defaults
    chosen = st.sidebar.multiselect("Compare recorded models", runs, default=defaults, format_func=metrics.label)
    if not chosen:
        st.info("Choose a recorded model.")
        return
    unit = "%" if metric == "MAPE" else UNITS
    held_out = [day for day in days if day.year >= 2025]
    st.caption(f"Held-out {held_out[0]} – {held_out[-1]} · {metric} ({unit}) · Lower is better. "
               "Observed targets only. MAPE uses observed counts ≥10. Fold scores are excluded.")
    table = metrics.scores(docs, chosen, metric)
    selected = table[table["Horizon (min)"] == h]
    st.dataframe(selected, hide_index=True)
    if selected[metric].notna().any():
        st.bar_chart(selected.set_index("Model")[metric], y_label=f"{metric} ({unit})")
    st.line_chart(table.pivot(index="Horizon (min)", columns="Model", values=metric), y_label=f"{metric} ({unit})")
    run = st.selectbox("Inspect model", chosen, format_func=metrics.label)
    history_runs = docs[run].get("runs", [run])
    for member in history_runs:
        history = pd.DataFrame(docs.get(member, {}).get("history", []))
        if {"epoch", "train_mae", "val_mae"} <= set(history):
            st.caption(f"{metrics.label(member)} · Training and validation MAE")
            st.line_chart(history.set_index("epoch")[["train_mae", "val_mae"]], y_label=UNITS)
        else:
            st.caption(f"{metrics.label(member)} · Training history N/A")
    st.subheader("Actual vs predicted")
    reason = models.availability(dataset, run, cfg)
    when = time_controls(days, dataset, cfg)
    site = st.selectbox("Sample SCATS site", list(sites.site_id))
    f = action("Load sample", (dataset, run, when), reason, lambda: models.forecast(dataset, run, when, cfg))
    if f is not None:
        st.caption(f"Single historical window · {metrics.label(run)} · {when} · observed targets shown where available")
        site_chart(f, site)


def render(build_deck):
    st.set_page_config(page_title="TBRGS", layout="wide")
    st.markdown("<style>.block-container{padding-top:2rem} [data-testid=stMetricValue]{font-size:1.9rem}</style>",
                unsafe_allow_html=True)
    page = st.sidebar.radio("Navigation", PAGES, key="page")
    st.title(page)
    try:
        cfg = ds.config()
        dataset = st.sidebar.selectbox("Dataset", list(ds.DATASETS), index=list(ds.DATASETS).index(cfg["dataset"]),
                                       key="dataset")
        sites, days = ds.catalog(dataset)
        st.sidebar.caption(f"{len(sites)} SCATS sites · {days[0]} – {days[-1]}")
        if dataset == ds.RESEARCH:
            st.caption("Historical replay · Melbourne local clock · Research dataset 2014–2026")
        if page == PAGES[0]:
            road_page(cfg, dataset, sites, days, build_deck)
        elif page == PAGES[1]:
            forecast_page(cfg, dataset, sites, days)
        else:
            evaluation_page(cfg, dataset, sites, days)
    except EXPECTED_ERRORS as exc:
        st.warning(str(exc))
