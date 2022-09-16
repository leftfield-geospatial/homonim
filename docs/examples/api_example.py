from pathlib import Path
from homonim import RasterFuse, RasterCompare, Model

# urls of source and reference test images
src_file = (
    'https://raw.githubusercontent.com/dugalh/homonim/main/'
    'tests/data/source/ngi_rgb_byte_1.tif'
)
ref_file = (
    'https://raw.githubusercontent.com/dugalh/homonim/main/'
    'tests/data/reference/sentinel2_b432_byte.tif'
)

# path to corrected file to create
corr_file = './corrected.tif'

# Correct src_file to surface reflectance by fusion with ref_file, using the
# `gain-blk-offset` model and a kernel of 5 x 5 pixels.
with RasterFuse(src_file, ref_file) as fuse:
    fuse.process(corr_file, Model.gain_blk_offset, (5, 5), overwrite=True)

# url of independent landsat reference for evaluation
cmp_ref_file = (
    'https://raw.githubusercontent.com/dugalh/homonim/main/'
    'tests/data/reference/landsat8_byte.tif'
)

# Compare source and corrected similarity with the independent reference,
# cmp_ref_file, giving an indication of the improvement in surface reflectance
# accuracy.
print('\nComparison key:\n' + RasterCompare.schema_table())
for cmp_src_file in [src_file, corr_file]:
    print(
        f'\nComparing {Path(cmp_src_file).name} with '
        f'{Path(cmp_ref_file).name}:'
    )
    with RasterCompare(cmp_src_file, cmp_ref_file) as compare:
        cmp_stats = compare.process()
        print(compare.stats_table(cmp_stats))
