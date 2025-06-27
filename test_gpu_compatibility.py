#!/usr/bin/env python3
"""
Test script to check GPU compatibility for Chatterbox TTS compilation.
This script helps diagnose compilation issues and provides guidance.
"""

import torch
from src.chatterbox import ChatterboxTTS


def test_gpu_compatibility():
    """Test GPU compatibility and provide guidance."""
    
    print("=== GPU Compatibility Test for Chatterbox TTS ===\n")
    
    # Check basic CUDA availability
    print("1. Basic CUDA Information:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            capability = torch.cuda.get_device_capability(i)
            print(f"   GPU {i}: {device_name} (CUDA Capability {capability[0]}.{capability[1]})")
    
    print("\n2. Loading Chatterbox TTS for compatibility check...")
    
    try:
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = ChatterboxTTS.from_pretrained(device)
        
        # Check compatibility
        compatibility = tts.check_gpu_compatibility()
        
        print("\n3. Compatibility Analysis:")
        print(f"   CUDA Available: {compatibility['cuda_available']}")
        
        if compatibility['cuda_available']:
            print(f"   Device Name: {compatibility['device_name']}")
            print(f"   CUDA Capability: {compatibility['cuda_capability'][0]}.{compatibility['cuda_capability'][1]}")
            print(f"   Supported Backends: {compatibility['supported_backends']}")
            print(f"   Recommended Backend: {compatibility['recommended_backend']}")
            
            # Provide guidance based on capability
            cuda_major = compatibility['cuda_capability'][0]
            
            if cuda_major >= 8:
                print("\n4. Performance Guidance:")
                print("   ✅ Excellent! Your GPU supports all compilation backends.")
                print("   🚀 Use 'inductor' backend for maximum performance.")
                print("   📈 Expected improvement: 30-40% faster inference")
                
            elif cuda_major >= 7:
                print("\n4. Performance Guidance:")
                print("   ✅ Good! Your GPU supports most compilation backends.")
                print("   🚀 Use 'inductor' backend for good performance.")
                print("   📈 Expected improvement: 20-30% faster inference")
                
            elif cuda_major >= 6:
                print("\n4. Performance Guidance:")
                print("   ⚠️  Limited! Your GPU has limited compilation support.")
                print("   🔧 Use 'aot_eager' backend for basic optimization.")
                print("   📈 Expected improvement: 10-20% faster inference")
                print("   💡 Consider upgrading to a newer GPU for better performance.")
                
            else:
                print("\n4. Performance Guidance:")
                print("   ❌ Not supported! Your GPU is too old for compilation.")
                print("   🔧 Will fall back to uncompiled model.")
                print("   📈 No performance improvement expected.")
                print("   💡 Consider upgrading to a newer GPU.")
        
        else:
            print("\n4. Performance Guidance:")
            print("   ❌ No CUDA available. Running on CPU.")
            print("   🔧 Limited compilation benefits on CPU.")
            print("   📈 Minimal performance improvement expected.")
        
        # Test auto-configuration
        print("\n5. Testing Auto-Configuration:")
        try:
            tts.auto_configure_t3_compilation()
            compilation_info = tts.get_t3_compilation_info()
            print(f"   Auto-configuration successful: {compilation_info}")
        except Exception as e:
            print(f"   Auto-configuration failed: {e}")
        
        print("\n6. Recommendations:")
        if compatibility['cuda_available']:
            cuda_major = compatibility['cuda_capability'][0]
            
            if cuda_major >= 7:
                print("   ✅ Your setup is optimal for compilation optimization.")
                print("   🎯 Use the default settings for best performance.")
                
            elif cuda_major >= 6:
                print("   ⚠️  Your setup will work but with limited optimization.")
                print("   🎯 Use 'aot_eager' backend for stability.")
                
            else:
                print("   ❌ Your setup is not suitable for compilation optimization.")
                print("   🎯 Consider running without compilation or upgrading hardware.")
        else:
            print("   ❌ No GPU acceleration available.")
            print("   🎯 Consider using a GPU-enabled system for better performance.")
        
    except Exception as e:
        print(f"\n❌ Error during compatibility check: {e}")
        print("   This might indicate an installation or configuration issue.")


def test_compilation_fallback():
    """Test compilation fallback mechanisms."""
    
    print("\n=== Compilation Fallback Test ===\n")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = ChatterboxTTS.from_pretrained(device)
        
        print("1. Testing fallback mechanisms...")
        
        # Try different backends
        backends_to_test = ["inductor", "aot_eager", "aot_ts"]
        
        for backend in backends_to_test:
            print(f"\n   Testing backend: {backend}")
            try:
                tts.configure_t3_compilation(backend=backend)
                compilation_info = tts.get_t3_compilation_info()
                print(f"   ✅ Success: {compilation_info}")
                break
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                continue
        
        print("\n2. Fallback test completed.")
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")


if __name__ == "__main__":
    print("Chatterbox TTS GPU Compatibility Test")
    print("=" * 50)
    
    # Run compatibility test
    test_gpu_compatibility()
    
    # Run fallback test
    test_compilation_fallback()
    
    print("\n" + "=" * 50)
    print("Compatibility test completed!")
    print("\nNext steps:")
    print("1. If compatibility is good, run: python test_compilation.py")
    print("2. If issues persist, check PyTorch and CUDA installation")
    print("3. For older GPUs, consider using 'aot_eager' backend") 