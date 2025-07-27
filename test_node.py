#!/usr/bin/env python3
"""
Test script to verify the FluxLogoTransferNode can be imported correctly
"""

def test_import():
    """Test importing the node"""
    try:
        from flux_logo_transfer import FluxLogoTransferNode
        print("[OK] FluxLogoTransferNode imported successfully")
        
        # Test node instantiation
        node = FluxLogoTransferNode()
        print("[OK] FluxLogoTransferNode instantiated successfully")
        
        # Test INPUT_TYPES
        inputs = FluxLogoTransferNode.INPUT_TYPES()
        print("[OK] INPUT_TYPES() works correctly")
        print(f"   Required inputs: {list(inputs['required'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def test_init_py():
    """Test the __init__.py registration"""
    try:
        import __init__
        print("[OK] __init__.py imported successfully")
        
        if hasattr(__init__, 'NODE_CLASS_MAPPINGS'):
            print("[OK] NODE_CLASS_MAPPINGS found")
            print(f"   Registered nodes: {list(__init__.NODE_CLASS_MAPPINGS.keys())}")
        else:
            print("[ERROR] NODE_CLASS_MAPPINGS not found")
            
        if hasattr(__init__, 'NODE_DISPLAY_NAME_MAPPINGS'):
            print("[OK] NODE_DISPLAY_NAME_MAPPINGS found")
            print(f"   Display names: {list(__init__.NODE_DISPLAY_NAME_MAPPINGS.values())}")
        else:
            print("[ERROR] NODE_DISPLAY_NAME_MAPPINGS not found")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] __init__.py test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Flux Logo Transfer Node...")
    print("=" * 50)
    
    success1 = test_import()
    print()
    success2 = test_init_py()
    
    print()
    print("=" * 50)
    if success1 and success2:
        print("[SUCCESS] All tests passed! Node should work in ComfyUI")
        print()
        print("Installation steps:")
        print("1. Copy all files to: /workspace/ComfyUI/custom_nodes/logoTransfer/")
        print("2. Install: pip install scipy")
        print("3. Restart ComfyUI")
        print("4. Search for: 'Flux Logo Transfer'")
    else:
        print("[FAILED] Tests failed. Check errors above.")