import matplotlib

# Important for headless servers / HPC nodes
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader


# ============================================================
# Output path
# ============================================================
base_dir = Path("/data/project/sleep_ENIGMA_insomnia/Codes/Results_making")
out_dir = base_dir / "geo_figs"
out_dir.mkdir(parents=True, exist_ok=True)

out_file = out_dir / "map_global_sites.svg"


# ============================================================
# City data
# ============================================================
cities = {
    "City": [
        "Seoul",
        "Chongqing",
        "Kermanshah",
        "Freiburg",
        "Utah",
        "Karolinska",
    ],
    "Country": [
        "South Korea",
        "China",
        "Iran",
        "Germany",
        "United States of America",
        "Sweden",
    ],
    "Latitude": [
        37.5665,   # Seoul
        29.5630,   # Chongqing
        34.3277,   # Kermanshah
        47.9990,   # Freiburg
        40.7608,   # Salt Lake City, Utah proxy
        59.3498,   # Karolinska Institutet / Stockholm
    ],
    "Longitude": [
        126.9780,   # Seoul
        106.5516,   # Chongqing
        47.0778,    # Kermanshah
        7.8421,     # Freiburg
        -111.8910,  # Salt Lake City, Utah proxy
        18.0686,    # Karolinska / Stockholm
    ],
}

target_countries = {
    "South Korea",
    "China",
    "Iran",
    "Germany",
    "United States of America",
    "Sweden",
}


# ============================================================
# Sanity checks
# ============================================================
n_city = len(cities["City"])
assert len(cities["Country"]) == n_city
assert len(cities["Latitude"]) == n_city
assert len(cities["Longitude"]) == n_city


# ============================================================
# Figure
# ============================================================
fig = plt.figure(figsize=(12, 7), dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

ax.set_global()

# Background
ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
ax.add_feature(cfeature.LAND, facecolor="#F2F2F2", zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="gray", zorder=2)
ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray", zorder=2)


# ============================================================
# Highlight countries
# ============================================================
countries_shp = shpreader.natural_earth(
    resolution="50m",
    category="cultural",
    name="admin_0_countries",
)

found_countries = set()

for country in shpreader.Reader(countries_shp).records():
    country_name = country.attributes["NAME"]

    if country_name in target_countries:
        found_countries.add(country_name)

        ax.add_geometries(
            [country.geometry],
            crs=ccrs.PlateCarree(),
            facecolor="#A1D99B",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

missing = target_countries - found_countries
if missing:
    print(f"Warning: these target countries were not found in Natural Earth: {missing}")


# ============================================================
# Plot city markers and labels
# ============================================================
label_offsets = {
    "Seoul": (5, 1.5),
    "Chongqing": (5, -2.5),
    "Kermanshah": (5, 1.5),
    "Freiburg": (5, -2.0),
    "Utah": (5, 1.5),
    "Karolinska": (5, 1.5),
}

for city, lat, lon in zip(
    cities["City"],
    cities["Latitude"],
    cities["Longitude"],
):
    ax.plot(
        lon,
        lat,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="darkgreen",
        markeredgewidth=2,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    dx, dy = label_offsets.get(city, (4, 1.5))

    ax.text(
        lon + dx,
        lat + dy,
        city,
        transform=ccrs.PlateCarree(),
        fontsize=9,
        ha="left",
        va="center",
        color="black",
        zorder=6,
    )


# Clean layout
ax.set_frame_on(False)

plt.tight_layout()

# Save
fig.savefig(out_file, format="svg", bbox_inches="tight")
plt.close(fig)

print(f"Saved figure to: {out_file}")
print(f"File exists: {out_file.exists()}")
print(f"File size: {out_file.stat().st_size if out_file.exists() else 0} bytes")