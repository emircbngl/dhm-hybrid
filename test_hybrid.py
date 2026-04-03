import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.julia_bridge import get_julia_core_module
from core.offaxis import extract_complex_field_offaxis, OffAxisParams
from core.reconstruction import propagate_asm, ReconstructionParams
from core.autofocus import autofocus_zscan, FocusMetric

def test_integration():
    print("Testing Julia Bridge Integration...")
    jl_core = get_julia_core_module()
    if jl_core is None:
        print("FAIL: Julia CoreModule not loaded.")
        sys.exit(1)
        
    print("SUCCESS: Julia CoreModule loaded.")
    
    # 1. Test OffAxis
    print("\nTesting offaxis extraction...")
    dummy_holo = np.random.rand(512, 512).astype(np.float32)
    params_offaxis = OffAxisParams(radius=20, dc_exclusion_radius=10)
    try:
        field, center = extract_complex_field_offaxis(dummy_holo, params_offaxis)
        print(f"SUCCESS: Offaxis extraction worked. Center: {center}, Field Shape: {field.shape}")
    except Exception as e:
        print(f"FAIL: Offaxis extraction threw an error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 2. Test Propagation (Reconstruction)
    print("\nTesting ASM propagation...")
    params_recon = ReconstructionParams(wavelength_m=633e-9, pixel_size_m=5e-6, z_m=0.1)
    try:
        recon_field = propagate_asm(field, params_recon)
        print(f"SUCCESS: ASM propagation worked. Field Shape: {recon_field.shape}")
    except Exception as e:
        print(f"FAIL: ASM propagation threw an error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 3. Test Autofocus
    print("\nTesting Autofocus Z-Scan...")
    z_values = np.linspace(0.09, 0.11, 5).tolist()
    from core.reconstruction import ReconstructionMethod
    try:
        result = autofocus_zscan(
            field=field, 
            base_params=params_recon, 
            z_values_m=z_values, 
            method=ReconstructionMethod.ASM, 
            metric=FocusMetric.TOTAL_VARIATION
        )
        print(f"SUCCESS: Autofocus Z-Scan worked. Best Z: {result.best_z_m}")
    except Exception as e:
        print(f"FAIL: Autofocus threw an error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\nALL TESTS PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    test_integration()
