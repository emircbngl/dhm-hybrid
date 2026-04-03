import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from core.offaxis import OffAxisParams, extract_complex_field_offaxis_debug

print("creating img")
img = np.random.rand(512, 512).astype(np.float32)
offaxis_params = OffAxisParams(radius=80)

print("Calling extract_complex_field_offaxis_debug")
try:
    fc, center, spec, mask = extract_complex_field_offaxis_debug(img, offaxis_params)
    print("Done calling extract_complex_field_offaxis_debug")
    print(center)
except Exception as e:
    import traceback
    traceback.print_exc()

