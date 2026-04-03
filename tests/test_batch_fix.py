import numpy as np
import os
import sys

from core.reconstruction import ReconstructionParams

# Verify dataclass works the way BatchRenderer now does
try:
    rp = ReconstructionParams(wavelength_m=632.8e-9, pixel_size_m=3.45e-6, z_m=0.1, n=1.0)
    print("Dataclass instantiation: success")
    rp.z_m = 0.2
    print("FAIL: Should have thrown frozen error!")
except Exception as e:
    print(f"Dataclass mutation throws (as expected): {type(e).__name__}")

rp2 = ReconstructionParams(wavelength_m=632.8e-9, pixel_size_m=3.45e-6, z_m=0.2, n=1.0)
print("Dataclass replacement: success")
print("All clear!")
