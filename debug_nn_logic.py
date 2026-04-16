import numpy as np
import sys

def debug_training_logic():
    print("Starting Training Logic Debug...")
    
    # Simulate data that might be problematic
    X_list = []
    labels = ["drone", "background"]
    
    # Add good data
    for _ in range(5):
        X_list.append(np.random.rand(1024).astype(np.float32))
    
    # What if something is weird? 
    # Let's try to find what could cause 'isnan' error on 'object' types
    
    try:
        X = np.array(X_list)
        print(f"X shape: {X.shape}, dtype: {X.dtype}")
        
        print("Testing np.isfinite(X)...")
        mask = np.isfinite(X)
        print("np.isfinite(X) OK")
        
        # Now simulate the 'object' dtype error
        X_obj = np.array(X_list, dtype=object)
        print(f"X_obj dtype: {X_obj.dtype}")
        try:
            np.isnan(X_obj)
        except Exception as e:
            print(f"Caught expected error on object array: {e}")
            
    except Exception as e:
        print(f"Logic failure: {e}")

if __name__ == "__main__":
    debug_training_logic()
