"""TBRGS GUI (Google Maps style). Run: streamlit run app.py
Uses the real dataset and trained models when they exist; otherwise it runs in demo mode.
Optional: put a Google Maps API key in config.json ("google_maps_api_key") to draw on the real Google Maps."""
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

from tbrgs.backend import load_system
from tbrgs.config import load_config

DIR = Path(__file__).parent
TIMES = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 15, 30, 45)]
PRESETS = {"Off-peak 03:00": "03:00", "Morning peak 08:00": "08:00", "Evening peak 17:00": "17:00"}
# Google Maps palette
BLUE, DARK_BLUE, GREY, WHITE = [66, 133, 244], [26, 86, 196], [154, 160, 166], [255, 255, 255]
GREEN, RED = [52, 168, 83], [234, 67, 53]
PX, MIDDLE, CENTER = '"pixels"', '"middle"', '"center"'   # pydeck reads a bare string as a column name: quote literals
FONT = '"Roboto, Arial, sans-serif"'


# ---------------------------------------------------------------- map helpers
def _along(coords, frac):
    """Point at a fraction of the length of a polyline of [lon, lat] points."""
    seg = [math.dist(coords[i], coords[i + 1]) for i in range(len(coords) - 1)]
    target, acc = frac * (sum(seg) or 1.0), 0.0
    for i, s in enumerate(seg):
        if acc + s >= target:
            t = (target - acc) / s if s else 0.0
            return [coords[i][0] + t * (coords[i + 1][0] - coords[i][0]), coords[i][1] + t * (coords[i + 1][1] - coords[i][1])]
        acc += s
    return list(coords[-1])


def label_points(routes, selected):
    """One spot per route for its time label, chosen so that labels do not sit on top of each other."""
    order = [selected] + [i for i in range(len(routes)) if i != selected]
    placed = {}
    for i in order:
        for frac in (0.5, 0.35, 0.65, 0.25, 0.75, 0.15, 0.85):
            p = _along(routes[i]["coords"], frac)
            if all(math.dist(p, q) > 0.006 for q in placed.values()):
                break
        placed[i] = p
    return placed


def _path_layer(rows, color, width):
    return pdk.Layer("PathLayer", rows, get_path="path", get_color=color, get_width=width, width_units=PX,
                     width_min_pixels=width, width_max_pixels=width, joint_rounded=True, cap_rounded=True, pickable=True)


def build_deck(sites, o, d, routes, selected=0):
    """Google Maps style: selected route in blue with a darker edge, alternatives in grey with a time bubble,
    origin A (green) and destination B (red). Every layer carries a `label` column for the tooltip."""
    sites = sites.assign(label=sites.site_id.astype(str) + " " + sites["name"])
    od = sites[sites.site_id.isin([o, d])].copy()
    od["tag"] = od.site_id.map({o: "A", d: "B"})
    od["color"] = od.site_id.map({o: GREEN, d: RED})
    spots = label_points(routes, selected) if routes else {}
    rows = [{"path": r["coords"], "route": i, "label": f"Route {i + 1}: {r['minutes']:.0f} min, {r['km']:.1f} km",
             "time": f"{r['minutes']:.0f} min", "lon": spots[i][0], "lat": spots[i][1]} for i, r in enumerate(routes)]
    cols = ["path", "route", "label", "time", "lon", "lat"]
    alt = pd.DataFrame([r for r in rows if r["route"] != selected], columns=cols)
    sel = pd.DataFrame([r for r in rows if r["route"] == selected], columns=cols)
    layers = [
        pdk.Layer("ScatterplotLayer", sites, get_position="[lon, lat]", get_radius=40, radius_min_pixels=4,
                  get_fill_color=GREY, stroked=True, get_line_color=WHITE, line_width_min_pixels=1, pickable=True),
        _path_layer(alt, WHITE, 9), _path_layer(alt, GREY, 6),
        _path_layer(sel, DARK_BLUE, 11), _path_layer(sel, BLUE, 8),
        pdk.Layer("TextLayer", alt, get_position="[lon, lat]", get_text="time", get_size=14, get_color=[60, 64, 67],
                  get_text_anchor=MIDDLE, get_alignment_baseline=CENTER, font_family=FONT, font_weight=600,
                  background=True, get_background_color=[255, 255, 255, 240], background_padding=[8, 5]),
        pdk.Layer("TextLayer", sel, get_position="[lon, lat]", get_text="time", get_size=14, get_color=WHITE,
                  get_text_anchor=MIDDLE, get_alignment_baseline=CENTER, font_family=FONT, font_weight=600,
                  background=True, get_background_color=BLUE + [255], background_padding=[8, 5]),
        pdk.Layer("ScatterplotLayer", od, get_position="[lon, lat]", get_radius=100, radius_min_pixels=13,
                  get_fill_color="color", stroked=True, get_line_color=WHITE, line_width_min_pixels=3, pickable=True),
        pdk.Layer("TextLayer", od, get_position="[lon, lat]", get_text="tag", get_size=16, get_color=WHITE,
                  get_text_anchor=MIDDLE, get_alignment_baseline=CENTER, font_family=FONT, font_weight=700),
    ]
    # Fit the origin, the destination and every route; deck.gl uses 512 px tiles, so zoom = log2(200 / span)
    pts = od[["lon", "lat"]].values.tolist() + [c for r in routes for c in r["coords"]]
    lons, lats = [p[0] for p in pts], [p[1] for p in pts]
    lat0, lon0 = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
    span = max(max(lons) - min(lons), (max(lats) - min(lats)) / math.cos(math.radians(lat0)), 0.002)
    view = pdk.ViewState(latitude=lat0, longitude=lon0, zoom=math.log2(200 / span))
    return pdk.Deck(layers=layers, initial_view_state=view, map_style="road", tooltip={"text": "{label}"})


GOOGLE_HTML = """<div id="map" style="height:__H__px;width:100%;border-radius:12px"></div>
<script>
const D = __DATA__;
function bubble(text, sel) {
  const w = text.length * 8 + 20;
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + w + '" height="26"><rect x="1" y="1" rx="13" ry="13" width="' + (w - 2) +
    '" height="24" fill="' + (sel ? '#4285F4' : '#fff') + '" stroke="' + (sel ? '#1A56C4' : '#9AA0A6') + '"/><text x="' + (w / 2) +
    '" y="17" text-anchor="middle" font-family="Roboto,Arial,sans-serif" font-size="13" font-weight="600" fill="' +
    (sel ? '#fff' : '#3C4043') + '">' + text + '</text></svg>';
  return {url: 'data:image/svg+xml;charset=UTF-8,' + encodeURIComponent(svg), anchor: new google.maps.Point(w / 2, 13)};
}
function pin(p, color, letter) {
  return new google.maps.Marker({position: p, title: p.t, zIndex: 50, label: {text: letter, color: '#fff', fontWeight: '700'},
    icon: {path: google.maps.SymbolPath.CIRCLE, scale: 13, fillColor: color, fillOpacity: 1, strokeColor: '#fff', strokeWeight: 3}});
}
function initMap() {
  const map = new google.maps.Map(document.getElementById('map'),
    {mapTypeControl: false, streetViewControl: false, clickableIcons: false, gestureHandling: 'greedy'});
  const bounds = new google.maps.LatLngBounds();
  D.sites.forEach(s => new google.maps.Marker({map: map, position: s, title: s.t, zIndex: 1,
    icon: {path: google.maps.SymbolPath.CIRCLE, scale: 4, fillColor: '#9AA0A6', fillOpacity: 1, strokeColor: '#fff', strokeWeight: 1}}));
  const order = D.routes.map((r, i) => i).filter(i => i !== D.selected).concat(D.routes.length ? [D.selected] : []);
  order.forEach(i => {
    const r = D.routes[i], sel = i === D.selected;
    r.path.forEach(p => bounds.extend(p));
    new google.maps.Polyline({map: map, path: r.path, strokeColor: sel ? '#1A56C4' : '#fff', strokeWeight: sel ? 11 : 9, zIndex: sel ? 20 : 10});
    new google.maps.Polyline({map: map, path: r.path, strokeColor: sel ? '#4285F4' : '#9AA0A6', strokeWeight: sel ? 8 : 6, zIndex: sel ? 21 : 11});
    new google.maps.Marker({map: map, position: r.label, icon: bubble(r.time, sel), zIndex: sel ? 40 : 30, title: r.title});
  });
  pin(D.o, '#34A853', 'A').setMap(map); pin(D.d, '#EA4335', 'B').setMap(map);
  bounds.extend(D.o); bounds.extend(D.d);
  map.fitBounds(bounds, 60);
}
</script>
<script src="https://maps.googleapis.com/maps/api/js?key=__KEY__&callback=initMap" async defer></script>"""


def google_map_html(api_key, sites, o, d, routes, selected=0, height=520):
    """Real Google Maps (needs an API key): same routes, colours and time bubbles as the pydeck map."""
    pos = label_points(routes, selected) if routes else {}
    sites = sites.assign(t=sites.site_id.astype(str) + " " + sites["name"])
    at = lambda s: {"lat": float(sites.loc[sites.site_id == s, "lat"].iloc[0]), "lng": float(sites.loc[sites.site_id == s, "lon"].iloc[0]),
                    "t": str(sites.loc[sites.site_id == s, "t"].iloc[0])}
    data = {"sites": [{"lat": float(r.lat), "lng": float(r.lon), "t": r.t} for r in sites.itertuples()],
            "o": at(o), "d": at(d), "selected": selected,
            "routes": [{"path": [{"lat": c[1], "lng": c[0]} for c in r["coords"]], "time": f"{r['minutes']:.0f} min",
                        "label": {"lat": pos[i][1], "lng": pos[i][0]}, "title": f"Route {i + 1}: {r['minutes']:.0f} min, {r['km']:.1f} km"}
                       for i, r in enumerate(routes)]}
    return GOOGLE_HTML.replace("__H__", str(height)).replace("__DATA__", json.dumps(data)).replace("__KEY__", api_key)


# ---------------------------------------------------------------- app
@st.cache_resource
def get_system():
    return load_system(load_config(DIR / "config.json"), DIR)


def main():
    st.set_page_config(page_title="TBRGS", layout="wide")
    st.markdown("<style>.block-container{padding-top:2rem} [data-testid=stMetricValue]{font-size:1.9rem}</style>", unsafe_allow_html=True)
    cfg = load_config(DIR / "config.json")
    system, notice = get_system()
    sites = system.sites
    ids = sites.site_id.astype(int).tolist()
    labels = dict(zip(ids, sites.site_id.astype(str) + " - " + sites["name"]))
    st.session_state.setdefault("origin", cfg["default_origin"] if cfg["default_origin"] in ids else ids[0])
    st.session_state.setdefault("dest", cfg["default_destination"] if cfg["default_destination"] in ids else ids[-1])
    st.session_state.setdefault("time", "08:00")

    def swap():
        st.session_state.origin, st.session_state.dest = st.session_state.dest, st.session_state.origin

    def preset():
        if st.session_state.get("preset") in PRESETS:
            st.session_state.time = PRESETS[st.session_state.preset]

    st.title("TBRGS")
    st.caption("Traffic-based route guidance for Boroondara: routes are timed with the traffic flow predicted by the models.")
    if cfg.get("_warning"):
        st.warning(cfg["_warning"])
    if notice:
        st.warning(notice)

    with st.sidebar:
        st.header("Directions")
        o = st.selectbox("From (SCATS site)", ids, key="origin", format_func=labels.get)
        d = st.selectbox("To (SCATS site)", ids, key="dest", format_func=labels.get)
        st.button("Swap origin and destination", on_click=swap)
        day = st.selectbox("Day", system.days, index=len(system.days) - 1)
        st.segmented_control("Quick time", list(PRESETS), key="preset", on_change=preset, label_visibility="collapsed")
        time_s = st.select_slider("Time of day", TIMES, key="time")
        models = st.multiselect("Prediction models", cfg["available_models"], default=cfg["default_models"])
        k = st.slider("Number of routes", 1, 5, cfg["k_routes"])

    if o == d:
        st.error("Origin and destination must be different.")
        return
    if not models:
        st.info("Choose at least one prediction model.")
        return

    when = datetime.fromisoformat(f"{day}T{time_s}")
    results = {}
    with st.spinner("Predicting traffic and searching routes..."):
        for m in models:
            try:
                routes = system.route(o, d, m, when, k)
                if routes:
                    results[m] = routes
                else:
                    st.error(f"{m}: no route found between these sites.")
            except ValueError as e:
                st.error(f"{m}: {e}")

    left, center, right = st.columns([1.1, 2.7, 1.2])
    shown, picked = None, 0
    with left:
        st.subheader("Routes")
        st.caption(when.strftime("Leave on %a %d %b %Y at %H:%M"))
        if results:
            shown = st.radio("Model", list(results), horizontal=True)
            routes = results[shown]
            best = routes[0]["minutes"]
            picked = st.radio("Route", range(len(routes)), key=f"route-{shown}-{len(routes)}", label_visibility="collapsed",
                              format_func=lambda i: f"{routes[i]['minutes']:.0f} min ({routes[i]['km']:.1f} km)"
                                                    + ("  Fastest" if i == 0 else f"  +{routes[i]['minutes'] - best:.0f} min"))
            st.caption("Via SCATS sites " + " > ".join(map(str, routes[picked]["path"])))
    with center:
        shown_routes = results[shown] if shown else []
        key = cfg.get("google_maps_api_key", "")
        if key:
            components.html(google_map_html(key, sites, o, d, shown_routes, picked, 560), height=570)
        else:
            st.pydeck_chart(build_deck(sites, o, d, shown_routes, picked), height=560)
        st.caption("A = origin, B = destination, grey dots = other SCATS sites (hover for names).")
    with right:
        st.subheader("Summary")
        if results:
            cur = results[shown][picked]
            st.metric("Est. travel time", f"{cur['minutes']:.1f} min")
            st.metric("Distance", f"{cur['km']:.2f} km")
            st.metric("SCATS sites on route", len(cur["path"]))
            if len(results) > 1:
                st.caption("Fastest route per model")
                st.dataframe(pd.DataFrame([{"Model": m, "Min": round(r[0]["minutes"], 1)}
                                           for m, r in results.items()]), hide_index=True)


if __name__ == "__main__":
    main()
