from pathlib import Path
from homonim import RasterFuse, RasterCompare, Model

# set source and reference etc paths from test data
src_file = Path('tests/data/source/ngi_rgb_byte_2.tif')
ref_file = Path('tests/data/reference/sentinel2_b432_byte.tif')
cmp_ref_file = Path('tests/data/reference/landsat8_byte.tif')
corr_file = Path('tests/data/corrected/corrected_2.tif')

# correct src_file to surface reflectance by fusion with ref_file
with RasterFuse(src_file, ref_file) as fuse:
    fuse.process(corr_file, Model.gain_blk_offset, (5, 5), overwrite=True)

# evaluate the change in surface reflectance accuracy by comparing source
# (src_file) and corrected (corr_file) files with cmp_ref_file
print('\nComparison key:\n' + RasterCompare.schema_table())
for cmp_src_file in [src_file, corr_file]:
    print(f'\nComparing {cmp_src_file.name} with {cmp_ref_file.name}:')
    with RasterCompare(cmp_src_file, cmp_ref_file) as compare:
        cmp_stats = compare.process()
        print(compare.stats_table(cmp_stats))
