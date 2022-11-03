# [correct-start]
from homonim import RasterFuse, Model

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
# [correct-end]

# [compare-start]
from homonim import RasterCompare
# url of independent landsat reference for evaluation
cmp_ref_file = (
    'https://raw.githubusercontent.com/dugalh/homonim/main/'
    'tests/data/reference/landsat8_byte.tif'
)

# Compare source and corrected similarity with the independent reference,
# cmp_ref_file, giving an indication of the improvement in surface reflectance
# accuracy.
summ_dict = {}
for cmp_src_file, cmp_src_label in zip(
    [src_file, corr_file],
    ['Source', 'Corrected'],
):
    with RasterCompare(cmp_src_file, cmp_ref_file) as compare:
        stats_dict = compare.process()
        summ_dict[cmp_src_label] = stats_dict['Mean']

# print comparison tables
print(RasterCompare.schema_table())
print('\n\n' + RasterCompare.stats_table(summ_dict))
# [compare-end]
