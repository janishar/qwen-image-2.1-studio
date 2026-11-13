"""The CLI's render size: the ratio table, and one side following the aspect.

Run: uv run python tests/test_sizes.py  (web/static/app.js mirrors these rules for the page's readout)
"""

from PIL import Image

from qwen_image_2_1.generate import ASPECT_RATIOS, plan_size

# neither side: the ratio's size, or the pipeline's own for an edit with no ratio
assert plan_size("16:9", None, None, None) == ASPECT_RATIOS["16:9"] == (2752, 1536)
assert plan_size(None, Image.new("RGB", (1200, 1600)), None, None) == (None, None)
# one side: the other follows the ratio, in multiples of 32 within 256-4096
assert plan_size("16:9", None, 2048, None) == (2048, 1152)
assert plan_size("16:9", None, None, 1152) == (2048, 1152)
assert plan_size("4:3", None, 2400, None) == (2400, 1792)  # the table's own size back
assert plan_size(None, Image.new("RGB", (1200, 1600)), 1536, None) == (1536, 2048)  # an edit: the last input's aspect
assert plan_size("9:16", None, 4096, None) == (4096, 4096)  # 7282 clamped
# both sides: a free size, exactly as given, whatever the ratio
assert plan_size("16:9", None, 1024, 1024) == (1024, 1024)
print("ok")
