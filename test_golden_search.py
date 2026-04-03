import numpy as np

def focus_metric_simulation(z: float) -> float:
    # A simulated bell curve focus function peaking exactly at 0.1234
    best_z = 0.1234
    width = 0.05
    return np.exp(-((z - best_z) / width) ** 2)

def dummy_golden_search(z_min, z_max, tol=1e-6, maximize=True):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    resphi = 2.0 - phi
    
    a, b = float(z_min), float(z_max)
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    
    f1 = focus_metric_simulation(x1)
    f2 = focus_metric_simulation(x2)
    
    iters = 0
    while abs(b - a) > tol:
        iters += 1
        pick_left = (f1 > f2) if maximize else (f1 < f2)
        if pick_left:
            b, x2, f2 = x2, x1, f1
            x1 = a + resphi * (b - a)
            f1 = focus_metric_simulation(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = b - resphi * (b - a)
            f2 = focus_metric_simulation(x2)
            
    best_z = (a + b) / 2.0
    return best_z, iters

best_z, iters = dummy_golden_search(0.0, 0.5)
print(f"Simulated peak: 0.1234")
print(f"Algorithm Output: {best_z:.6f} in {iters} iterations")
print("SUCCESS!" if abs(best_z - 0.1234) < 1e-4 else "FAIL!")
