"""Plot aligned SCATS sites on OpenStreetMap to eyeball alignment."""
import json
import folium

sites = json.load(open("core/sites_aligned.json"))
m = folium.Map(location=[sum(s["lat"] for s in sites) / len(sites),
                         sum(s["lon"] for s in sites) / len(sites)],
               zoom_start=13, tiles="OpenStreetMap")

for s in sites:
    folium.CircleMarker(
        [s["lat"], s["lon"]], radius=6,
        color="red" if s.get("out_of_bounds") else "blue", fill=True,
        tooltip=str(s["site"]),
        popup=f'Site {s["site"]}<br>{s["lat"]:.5f}, {s["lon"]:.5f}',
    ).add_to(m)

folium.LatLngPopup().add_to(m)
m.save("core/coord_check.html")
print("Right click 'core/coord_check.html' -> Choose 'Open with Live Server' to view the map in your browser")
