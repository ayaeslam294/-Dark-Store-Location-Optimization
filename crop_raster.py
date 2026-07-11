import osmnx as ox
import rasterio
from rasterio.mask import mask

# 1. Your raw country-wide dataset path
RAW_EGYPT_RASTER = "C:\\Users\\ywsfy\\Downloads\\Project\\egy_pop_2025_CN_100m_R2025A_v1.tif"

# 2. The output file the prep script is looking for
CROPPED_NASR_CITY_RASTER = "nasr_city_constrained_population_100m.tif"

print("Fetching Nasr City boundary from OSM...")
nasr_city_boundary = ox.geocode_to_gdf("Nasr City, Cairo, Egypt")

print(f"Opening {RAW_EGYPT_RASTER} and clipping to Nasr City...")
try:
    with rasterio.open(RAW_EGYPT_RASTER) as src:
        # Align the CRS of the boundary with the raster
        nasr_city_boundary = nasr_city_boundary.to_crs(src.crs)
        shapes = [feature for feature in nasr_city_boundary.geometry]
        
        # Crop the raster to the boundary shape
        out_image, out_transform = mask(src, shapes, crop=True)
        out_meta = src.meta.copy()

        # Update metadata for the new, smaller file
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Save the new file
        with rasterio.open(CROPPED_NASR_CITY_RASTER, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"Success! Cropped file saved locally as: {CROPPED_NASR_CITY_RASTER}")
    print("You can now run your dark_store_data_prep.py execution block.")

except Exception as e:
    print(f"An error occurred: {e}")