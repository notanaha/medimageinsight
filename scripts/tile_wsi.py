"""Read tissue tiles out of a whole-slide image at a fixed microns-per-pixel resolution.

MedImageInsight accepts ordinary PNGs and states no required magnification, so `target_mpp`
here is *our* experiment setting, not a model requirement. It is fixed anyway, because tiles
read at different magnifications are not comparable to each other: the same lesion sampled at
0.5 MPP and at 1.0 MPP produces different embeddings, and the slides in a cohort rarely share
a scanner.

A slide's pyramid levels are whatever the scanner wrote, so the level index alone tells you
nothing. This module reads the MPP from the slide metadata, picks the finest level that is
still at or below the target, and resizes the remainder.
"""

import numpy as np
from PIL import Image

WSI_SUFFIXES = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn", ".bif")

# A tile is read as a whole number of pixels and then resized, so the magnification it ends up
# carrying lands near the target rather than exactly on it. Rounding moves the read by at most
# half a pixel and `pick_level` keeps the read at least `tile_size / 1.01` wide, which bounds the
# drift at about 0.1%. Anything past this tolerance is a broken calculation, not a coarse slide.
MPP_TOLERANCE = 0.01


def read_base_mpp(slide):
    """Microns per pixel at level 0. No default: guessing this silently rescales everything."""
    # Spelled out rather than taken from openslide.PROPERTY_NAME_*, so this stays importable
    # (and testable) on a laptop that has no OpenSlide binaries.
    for key in ("openslide.mpp-x", "aperio.MPP", "openslide.mpp-y"):
        value = slide.properties.get(key)
        if value:
            return float(value)
    raise ValueError("Slide has no MPP metadata; magnification cannot be resolved.")


def pick_level(slide, base_mpp, target_mpp):
    """Coarsest pyramid level that is still at or finer than the target, so we only downsample."""
    level = 0
    for candidate in range(slide.level_count):
        if base_mpp * slide.level_downsamples[candidate] <= target_mpp * 1.01:
            level = candidate
    return level


def effective_mpp(base_mpp, level_downsample, read_size, tile_size):
    """Magnification the finished tile actually carries, after the whole-pixel read and the resize."""
    return base_mpp * level_downsample * read_size / tile_size


def otsu_threshold(values):
    """Otsu's split of a luminance histogram, written out so the job needs no OpenCV/skimage."""
    histogram, _ = np.histogram(values, bins=256, range=(0, 256))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    intensity = np.arange(256, dtype=np.float64)
    weight_background = np.cumsum(histogram)
    weight_foreground = total - weight_background
    mean_background = np.cumsum(histogram * intensity) / np.maximum(weight_background, 1)
    mean_total = (histogram * intensity).sum() / total
    mean_foreground = ((mean_total * total - np.cumsum(histogram * intensity))
                       / np.maximum(weight_foreground, 1))
    variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
    return float(np.argmax(variance))


def tissue_occupancy(slide, grid_shape, span):
    """Fraction of tissue in each grid cell, measured on a thumbnail rather than on real tiles.

    Deciding which tiles are worth reading costs one small decode this way, instead of decoding
    every candidate tile at full resolution first.
    """
    grid_x, grid_y = grid_shape
    width0, height0 = slide.level_dimensions[0]
    thumbnail = slide.get_thumbnail((min(grid_x * 8, 4096), min(grid_y * 8, 4096))).convert("L")
    luminance = np.asarray(thumbnail, dtype=np.uint8)
    tissue = luminance <= otsu_threshold(luminance)  # stained tissue is darker than the glass

    step_x = span * luminance.shape[1] / width0
    step_y = span * luminance.shape[0] / height0
    occupancy = np.zeros((grid_y, grid_x), dtype=np.float32)
    for row in range(grid_y):
        y0, y1 = int(row * step_y), max(int((row + 1) * step_y), int(row * step_y) + 1)
        for column in range(grid_x):
            x0, x1 = int(column * step_x), max(int((column + 1) * step_x), int(column * step_x) + 1)
            patch = tissue[y0:y1, x0:x1]
            occupancy[row, column] = patch.mean() if patch.size else 0.0
    return occupancy


def tile_slide(path, target_mpp=0.5, tile_size=512, occupancy_threshold=0.20, max_tiles=32):
    """Return (list of RGB tiles at target_mpp, metadata dict). Tiles stay in memory."""
    import openslide

    with openslide.OpenSlide(str(path)) as slide:
        base_mpp = read_base_mpp(slide)
        mpp_y = float(slide.properties.get("openslide.mpp-y", base_mpp))
        if not all(np.isfinite(value) and value > 0 for value in (base_mpp, mpp_y, target_mpp)):
            raise ValueError("MPP values must be finite and positive.")
        if not np.isclose(base_mpp, mpp_y, rtol=0.02):
            raise ValueError("Non-square pixels need a separately validated resampling step.")
        if base_mpp > target_mpp * 1.02:
            # Upsampling invents detail the scanner never captured.
            raise ValueError(f"Slide resolution {base_mpp:.4f} MPP is coarser than the "
                             f"{target_mpp} MPP target.")

        level = pick_level(slide, base_mpp, target_mpp)
        level_downsample = float(slide.level_downsamples[level])
        span = tile_size * target_mpp / base_mpp          # tile edge in level-0 pixels
        read_size = int(round(span / level_downsample))   # pixels actually decoded
        if not 1 <= read_size <= 8192:
            raise ValueError("Unsafe tile decode size.")

        achieved_mpp = effective_mpp(base_mpp, level_downsample, read_size, tile_size)
        if abs(achieved_mpp - target_mpp) > target_mpp * MPP_TOLERANCE:
            raise ValueError(f"Tiles would carry {achieved_mpp:.4f} MPP instead of the "
                             f"{target_mpp} MPP target, so they cannot be compared with the "
                             f"rest of the cohort.")

        width0, height0 = slide.level_dimensions[0]
        grid_x, grid_y = int(width0 // span), int(height0 // span)
        if not grid_x or not grid_y or grid_x * grid_y > 1_000_000:
            raise ValueError("Invalid or excessive tiling grid.")

        occupancy = tissue_occupancy(slide, (grid_x, grid_y), span)
        candidates = np.argwhere(occupancy >= occupancy_threshold)
        if not len(candidates):
            raise ValueError("No tile meets the tissue occupancy threshold.")

        # Evenly spaced through the tissue grid rather than the densest tiles. Picking by
        # occupancy would concentrate every slide's sample on its thickest region.
        chosen = np.linspace(0, len(candidates) - 1, min(len(candidates), max_tiles), dtype=int)

        tiles, coordinates = [], []
        for index in chosen:
            row, column = candidates[index]
            x0, y0 = int(round(column * span)), int(round(row * span))
            rgba = slide.read_region((x0, y0), level, (read_size, read_size)).convert("RGBA")
            rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
            tiles.append(rgb.resize((tile_size, tile_size), Image.Resampling.LANCZOS))
            coordinates.append({"x_level0": x0, "y_level0": y0,
                                "occupancy": float(occupancy[row, column])})

        metadata = {
            "base_mpp": base_mpp,
            "target_mpp": float(target_mpp),
            "read_level": level,
            "level_downsample": level_downsample,
            "read_size": read_size,
            "tile_size": int(tile_size),
            # Within MPP_TOLERANCE of target_mpp, checked above. Not equal to it: see the constant.
            "effective_mpp": achieved_mpp,
            "grid": [grid_x, grid_y],
            "candidate_tiles": int(len(candidates)),
            "selected_tiles": len(tiles),
            "sampling_fraction": len(tiles) / len(candidates),
            "sampling": "evenly_spaced_in_row_major_tissue_grid",
            "occupancy_threshold": float(occupancy_threshold),
            "tiles": coordinates,
        }
    return tiles, metadata
