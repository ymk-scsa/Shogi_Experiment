import torch
import sys
import os

# Root of Shogi_Experience
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from yugiwarabe.yugiwarabe import create_yugiwarabe

def test_advanced_model():
    input_ch = 120 # Design doc suggests around 100-120
    num_actions = 2187 # Standard for Shogi labels
    hidden_dim = 256
    
    model = create_yugiwarabe(input_ch, num_actions)
    
    # [B, T, C, H, W]
    B, T = 2, 16
    dummy_input = torch.randn(B, T, input_ch, 9, 9)
    
    # Optional piece_mask [B*T, 81, 81]
    dummy_piece_mask = torch.zeros(B * T, 81, 81)
    
    policy, value, aux = model(dummy_input, piece_mask=dummy_piece_mask)
    
    print("Policy shape:", policy.shape)
    print("Value shape:", value.shape)
    for k, v in aux.items():
        print(f"Aux {k} shape: {v.shape}")
    
    assert policy.shape == (B, num_actions)
    assert value.shape == (B, 1)
    assert len(aux) == 6
    
    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    print("Advanced Model test passed!")

if __name__ == "__main__":
    test_advanced_model()
