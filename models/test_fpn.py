import torch
from fpn import FPN

def test_fpn_output_shapes():
    dummy_features = [
        torch.randn(2, 128, 32, 32),   
        torch.randn(2, 256, 16, 16),   
        torch.randn(2, 512, 8, 8),
    ]

    fpn = FPN(in_channels_list=[128, 256, 512], out_channels=256)
    outputs = fpn(dummy_features)

    print(f"Number of input feature maps: {len(dummy_features)}")
    print(f"Number of output feature maps: {len(outputs)}")

    for i, (inp, out) in enumerate(zip(dummy_features, outputs)):
        print(f"Level {i}: input shape {tuple(inp.shape)} -> output shape {tuple(out.shape)}")
        
        assert out.shape[-2:] == inp.shape[-2:], f"Spatial size mismatch at level {i}"
        
        assert out.shape[1] == 256, f"Channel count not unified at level {i}"

    print("\nAll shape checks passed! FPN correctly fuses multi-scale features.")


if __name__ == "__main__":
    test_fpn_output_shapes()
