import pandas as pd
import geopandas
import matplotlib.pyplot as plt
from geodatasets import get_path

# %%
df = pd.DataFrame(
    {
        "City": ['Liege', 'Greifswald', 'Solna', 'Juelich', 'Rotterdam'],
        "Country": ['Belgium', 'Germany', 'Sweden', 'Germany', 'Netherlands'],
        "Latitude": [50.6326, 54.0866, 59.3566, 50.9210, 51.9225],
        "Longitude": [5.5797, 13.3996, 18.0229, 6.3618, 4.4792],
    }
)

gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
)

# %%
world = geopandas.read_file(get_path("naturalearth.land"))

plt.figure(figsize=(10, 5))
ax = world.clip([-5.0, 48.0, 30.0, 71.0]).plot(color="white", edgecolor="black")

# We can now plot our ``GeoDataFrame``.
gdf.plot(ax=ax, color="red")
plt.tight_layout()
plt.show()
plt.close()

# %%
# Set up the plot
plt.figure(figsize=(10, 5))

# Clip to the bounding box and plot the countries with a green fill and white borders
ax = world.clip([-5.0, 48.0, 30.0, 71.0]).plot(color="#a1d99b", edgecolor="white", linewidth=1.5)

# Plot the cities with red points
gdf.plot(ax=ax, color="red", markersize=50)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
plt.close()
