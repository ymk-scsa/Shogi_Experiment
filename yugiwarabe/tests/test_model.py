import torch
import sys
import os

# Root of Shogi_Experience
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from yugiwarabe.model import create_yugiwarabe

def test_model():
    input_ch = 46 # Standard features
    num_actions = 2187 # Standard for Shogi labels
    model = create_yugiwarabe(input_ch, num_actions)
    
    # [B, T, C, H, W]
    dummy_input = torch.randn(2, 16, input_ch, 9, 9)
    policy, value = model(dummy_input)
    
    print("Policy shape:", policy.shape)
    print("Value shape:", value.shape)
    
    assert policy.shape == (2, num_actions)
    assert value.shape == (2, 1)
    print("Model test passed!")

if __name__ == "__main__":
    test_model()
