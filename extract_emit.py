import ee
import geemap
import sys
import json

def get_emit_image(lat, lon, start_date='2025-01-01', end_date='2025-12-31'):
    """
    Retrieves the EMIT hyperspectral image collection, filters by location and date,
    and returns a median composite.
    """
    point = ee.Geometry.Point([lon, lat])

    # NASA EMIT L2A Reflectance Collection
    collection = ee.ImageCollection("NASA/EMIT/L2A/RFL") \
        .filterBounds(point) \
        .filterDate(start_date, end_date)

    # Check if collection is empty (handled in main execution flow usually,
    # but here we return the median)
    image = collection.median()
    return image, point

def process_emit(lat, lon, start_date='2025-01-01', end_date='2025-12-31'):
    try:
        ee.Initialize()
    except Exception as e:
        return {"error": f"Error initializing Earth Engine: {e}"}

    # 1. Get Data
    image, point = get_emit_image(lat, lon, start_date, end_date)

    # Region of Interest: 200x200 pixels centered on point.
    region = point.buffer(6000).bounds()

    # 2. Extract Wavelengths and Band Names
    try:
        # Fetch metadata from the collection to get band info
        col = ee.ImageCollection("NASA/EMIT/L2A/RFL").filterBounds(point).filterDate(start_date, end_date)

        # We need at least one image to get metadata
        ref_image = col.first()

        # Helper to get info (client side fetch)
        info = ref_image.getInfo()
        if not info:
            return {"error": "No data found for this location/date."}

        bands = info.get('bands', [])
        properties = info.get('properties', {})
        wavelengths = properties.get('wavelengths')

        if not wavelengths:
            return {"error": "Could not extract wavelengths from metadata. Property 'wavelengths' missing."}

        band_names = [b['id'] for b in bands]

    except Exception as e:
        return {"error": f"Metadata extraction failed: {e}"}

    # Map bands to wavelengths
    def find_nearest_idx(target):
        return min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - target))

    idx_red = find_nearest_idx(665)
    idx_nir = find_nearest_idx(842)

    band_red = band_names[idx_red]
    band_nir = band_names[idx_nir]

    # Identify SWIR bands (2000-2500 nm)
    swir_indices = [i for i, w in enumerate(wavelengths) if 2000 <= w <= 2500]
    if not swir_indices:
        return {"error": "No bands found in 2000-2500nm range."}

    swir_bands = [band_names[i] for i in swir_indices]
    swir_wavelengths = [wavelengths[i] for i in swir_indices]

    # 3. Calculate NDVI
    ndvi = image.normalizedDifference([band_nir, band_red]).rename('NDVI')

    # 4. SWIR Processing (Continuum Removal & Feature Extraction)
    # Convert SWIR bands to an Array Image
    swir_image = image.select(swir_bands).toArray()

    # Continuum Removal: Linear Baseline
    y_start = swir_image.arrayGet([0])
    y_end = swir_image.arrayGet([len(swir_bands) - 1])

    x_start = swir_wavelengths[0]
    x_end = swir_wavelengths[-1]

    # Linear interpolation: y = mx + c
    slope = (y_end.subtract(y_start)).divide(ee.Number(x_end - x_start))

    # Construct continuum array using map server-side
    def calculate_continuum(w):
        w = ee.Number(w)
        # y_continuum is an Image (y_start + m*(w-x_start))
        return y_start.add(slope.multiply(w.subtract(x_start)))

    continuum_list = ee.List(swir_wavelengths).map(calculate_continuum)

    # Convert List of Images to ImageCollection -> toBands -> toArray
    # This creates an Array Image where each pixel is the array of continuum values
    continuum_array = ee.ImageCollection.fromImages(continuum_list).toBands().toArray()

    # Continuum Removed Reflectance = Original / Continuum
    # Result: 1.0 at edges, < 1.0 in absorption features
    cr_image = swir_image.divide(continuum_array)

    # Feature Extraction
    # 1. Depth (1 - reflectance)
    # Find minimum CR value
    min_cr = cr_image.arrayReduce(ee.Reducer.min(), [0]).arrayGet([0])
    depth = ee.Image(1).subtract(min_cr).rename('depth')

    # 2. Min Wavelength
    # Create an image where every pixel is the wavelengths array
    wavelengths_image = ee.Image(ee.Array(swir_wavelengths))

    # Sort wavelengths by CR values
    # arraySort(keys): sorts the array elements based on keys array
    min_wavelength = wavelengths_image.arraySort(cr_image).arrayGet([0]).rename('min_wavelength')

    # 3. Width (FWHM)
    # Count bands where CR < (1 - depth/2)
    threshold = ee.Image(1).subtract(depth.divide(2))
    is_feature = cr_image.lt(threshold)
    count_feature = is_feature.arrayReduce(ee.Reducer.sum(), [0]).arrayGet([0])

    avg_spacing = (x_end - x_start) / (len(swir_bands) - 1)
    width = count_feature.multiply(avg_spacing).rename('width')

    # 5. Correction
    # corrected_depth = depth / (1 - NDVI) for pixels with NDVI <= 0.7
    valid_mask = ndvi.lte(0.7)

    corrected_depth = depth.divide(ee.Image(1).subtract(ndvi)).rename('corrected_depth')

    # Pack result image
    # Note: We return an image with all calculated bands, masked by validity
    final_image = ee.Image.cat([min_wavelength, depth, width, corrected_depth]) \
        .updateMask(valid_mask)

    # 6. Output Generation
    # Return JSON with center pixel stats and region mean

    stats_mean = final_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=60,
        maxPixels=1e6
    )

    center_stats = final_image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=60
    )

    # Retrieve Center Pixel Full Spectrum (Original)
    center_spectrum = image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=60
    )

    output = {
        "spatial_mean": stats_mean.getInfo(),
        "center_pixel": center_stats.getInfo(),
        "full_spectrum": center_spectrum.getInfo()
    }

    return output, final_image

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_emit.py <lat> <lon>")
        sys.exit(1)

    try:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
    except ValueError:
        print("Error: Latitude and Longitude must be numbers.")
        sys.exit(1)

    result_json, result_image = process_emit(lat, lon)

    # Output JSON to stdout
    print(json.dumps(result_json, indent=2))
