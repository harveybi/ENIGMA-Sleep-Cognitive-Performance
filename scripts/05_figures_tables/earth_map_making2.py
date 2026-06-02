from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %%
# City data
cities = {
    "City": ['Liege', 'Greifswald', 'Solna', 'Juelich', 'Rotterdam'],
    "Country": ['Belgium', 'Germany', 'Sweden', 'Germany', 'Netherlands'],
    "Latitude": [50.6326, 54.0866, 59.3566, 50.9210, 51.9225],
    "Longitude": [5.5797, 13.3996, 18.0229, 6.3618, 4.4792],
}

# Increase figsize and dpi for higher resolution and larger size
fig = plt.figure(dpi=300)  # Larger size and even higher resolution
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([2, 20, 48, 62], crs=ccrs.PlateCarree())

# Put a background image on for nice sea rendering.
# ax.stock_img()

# Create a feature for country boundaries at 1:50m from Natural Earth.
countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_countries',
    scale='50m',
    facecolor='none')

# Add the country boundaries.
ax.add_feature(countries, edgecolor='black', linewidth=0.5)  # gray, 1.0

# Add land feature, overriding the default negative zorder so it shows
# above the background image.
ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k')

# Plot the cities with white dots and add a larger circle outline in green
for i in range(len(cities['City'])):
    ax.plot(cities['Longitude'][i], cities['Latitude'][i], marker='o', color='white', markersize=8,
            markeredgewidth=1.5, markeredgecolor='green', transform=ccrs.PlateCarree())
    # Optional: Add labels next to the dots if needed
    # ax.text(cities['Longitude'][i] + 0.5, cities['Latitude'][i] + 0.5, cities['City'][i],
    #         fontsize=9, transform=ccrs.PlateCarree())

plt.tight_layout()
plt.show()
plt.close()

# %%
# City data
cities = {
    "City": ['Liege', 'Greifswald', 'Solna', 'Juelich', 'Rotterdam'],
    "Country": ['Belgium', 'Germany', 'Sweden', 'Germany', 'Netherlands'],
    "Latitude": [50.6326, 54.0866, 59.3566, 50.9210, 51.9225],
    "Longitude": [5.5797, 13.3996, 18.0229, 6.3618, 4.4792],
}

# Increase figsize and dpi for higher resolution and larger size
fig = plt.figure(dpi=300)  # Larger size and even higher resolution
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([2, 20, 48, 62], crs=ccrs.PlateCarree())

# Put a background image on for nice sea rendering.
# ax.stock_img()

# Create a feature for country boundaries at 1:50m from Natural Earth.
countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_countries',
    scale='50m',
    facecolor='none')

# Add the country boundaries.
ax.add_feature(countries, zorder=2, edgecolor='black', linewidth=0.5)  # gray, 1.0

# Add land feature, overriding the default negative zorder so it shows above the background image.
# ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='#d5e1b4')
ax.add_feature(cfeature.LAND, zorder=1, edgecolor='black', linewidth=0.5, facecolor='#d5e1b4')

# Plot the cities with white dots and add a larger circle outline in green
for i in range(len(cities['City'])):
    ax.plot(cities['Longitude'][i], cities['Latitude'][i], marker='o', color='white', markersize=8,
            markeredgewidth=1.5, markeredgecolor='green', transform=ccrs.PlateCarree())
    # Optional: Add labels next to the dots if needed
    # ax.text(cities['Longitude'][i] + 0.5, cities['Latitude'][i] + 0.5, cities['City'][i],
    #         fontsize=9, transform=ccrs.PlateCarree())

plt.tight_layout()
plt.show()
plt.close()

# %%
import cartopy.io.shapereader as shpreader

# City data
cities = {
    "City": ['Liege', 'Greifswald', 'Solna', 'Juelich', 'Rotterdam'],
    "Country": ['Belgium', 'Germany', 'Sweden', 'Germany', 'Netherlands'],
    "Latitude": [50.6326, 54.0866, 59.3566, 50.9210, 51.9225],
    "Longitude": [5.5797, 13.3996, 18.0229, 6.3618, 4.4792],
}

# Countries of interest
target_countries = {'Belgium', 'Germany', 'Sweden', 'Netherlands'}

# Load Natural Earth countries data using Cartopy's shapefile reader
shapename = 'admin_0_countries'
countries_shp = shpreader.natural_earth(resolution='110m',
                                        category='cultural', name=shapename)

# Create the plot
fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([2, 20, 48, 62], crs=ccrs.PlateCarree())

# Function to colorize countries based on whether they are in the target list
def colorize_country(country_name):
    if country_name in target_countries:
        return '#a1d99b'  # green for target countries
    else:
        return '#f0f0f0'  # light grey for other countries

# Add geometries with custom coloring
for country in shpreader.Reader(countries_shp).records():
    facecolor = colorize_country(country.attributes['NAME'])
    ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                      facecolor=facecolor, edgecolor='black', linewidth=0.5)

# Plot the cities with white dots and add a larger circle outline in green
for i in range(len(cities['City'])):
    ax.plot(cities['Longitude'][i], cities['Latitude'][i], marker='o', color='white', markersize=8,
            markeredgewidth=1.5, markeredgecolor='green', transform=ccrs.PlateCarree())

# Remove axes for a cleaner look
ax.set_axis_off()

plt.tight_layout()
plt.show()
plt.close()

# %%
# City data
cities = {
    "City": ['Liege', 'Greifswald', 'Solna', 'Juelich', 'Rotterdam'],
    "Country": ['Belgium', 'Germany', 'Sweden', 'Germany', 'Netherlands'],
    "Latitude": [50.6326, 54.0866, 59.3566, 50.9210, 51.9225],
    "Longitude": [5.5797, 13.3996, 18.0229, 6.3618, 4.4792],
}

# Countries of interest
target_countries = {'Belgium', 'Germany', 'Sweden', 'Netherlands'}

# Increase figsize and dpi for higher resolution and larger size
fig = plt.figure(dpi=300)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([2, 20, 48, 62], crs=ccrs.PlateCarree())

# Load Natural Earth countries data using Cartopy's shapefile reader
shapename = 'admin_0_countries'
countries_shp = shpreader.natural_earth(resolution='50m',
                                        category='cultural', name=shapename)

# Function to colorize countries based on whether they are in the target list
def colorize_country(country_name):
    if country_name in target_countries:
        return '#a1d99b'  # green for target countries
    else:
        return '#f0f0f0'  # light grey for other countries

# Add geometries with custom coloring
for country in shpreader.Reader(countries_shp).records():
    facecolor = colorize_country(country.attributes['NAME'])
    ax.add_geometries([country.geometry], ccrs.PlateCarree(),
                      facecolor=facecolor, edgecolor='black', linewidth=0.5)

# Plot the cities with white dots and add a larger circle outline in green
for i in range(len(cities['City'])):
    ax.plot(cities['Longitude'][i], cities['Latitude'][i], marker='o', color='white', markersize=8,
            markeredgewidth=1.5, markeredgecolor='green', transform=ccrs.PlateCarree())

# Remove axes for a cleaner look
ax.set_axis_off()

plt.tight_layout()
plt.show()
plt.close()