import torch
import torch.nn as nn
import torch.nn.functional as F

class DisparityUpsampler(nn.Module):
    def __init__(self, image_dim, feature_dim, scale):
        super().__init__()
        self.scale = scale

        # Extract features at 1/4 scale
        self.encode = nn.Sequential(
            nn.Conv2d(image_dim, 256, kernel_size=scale, stride=scale, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv_disp_feat = nn.Sequential(
            nn.Conv2d(1 + feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, scale**2 * 9, kernel_size=1)
        )

    def detect_nans_in_sequential(self, sequential_module, input_tensor):
        x = input_tensor.clone()
        results = {"input_has_nan": torch.isnan(x).any().item()}
        
        if results["input_has_nan"]:
            print(f"Input tensor already contains NaNs!")
            return results
        
        # Process each layer individually
        for i, layer in enumerate(sequential_module):
            layer_name = f"Layer {i}: {layer.__class__.__name__}"
            print(f"Processing {layer_name}")
            
            # Forward pass through this layer only
            try:
                x = layer(x)
                
                # Check for NaNs
                has_nan = torch.isnan(x).any().item()
                results[layer_name] = {
                    "has_nan": has_nan,
                    "min": x.min().item() if not has_nan else "NaN",
                    "max": x.max().item() if not has_nan else "NaN",
                    "mean": x.mean().item() if not has_nan else "NaN"
                }
                
                if has_nan:
                    print(f"NaNs first detected after {layer_name}")
                    results["first_nan_layer"] = i
                    results["first_nan_layer_name"] = layer_name
                    break
                    
            except Exception as e:
                print(f"Error in {layer_name}: {e}")
                results[layer_name] = {"error": str(e)}
                break
        
        return results
    
    def forward(self, patch, disp):
        B, _, H, W = disp.shape

        # Encoder
        feat = self.encode(patch)

        # print(f'Patch: {patch.shape}')
        # print(f'Disp: {disp.shape}')
        # print(f'Feature: {feat.shape}')
        
        # Concatenate disparity and features along channel dimension
        x = torch.cat([disp, feat], dim=1)

        # # Debug where nan's are coming from
        # results = self.detect_nans_in_sequential(self.conv_disp_feat, x)
        
        # Predict mask and normalize over neighborhood
        raw_mask = self.conv_disp_feat(x)
        mask = raw_mask.view(B, 1, 9, self.scale, self.scale, H, W)
        mask = torch.softmax(mask, dim=2)

        # Extract 3x3 neighborhoods from disparity
        unfolded_disp = F.unfold(disp, kernel_size=3, padding=1)
        unfolded_disp = unfolded_disp.view(B, 1, 9, 1, 1, H, W)

        # Apply learned mask as weights for convex combination
        up_disp = (mask * unfolded_disp).sum(dim=2)
        up_disp= up_disp.permute(0, 1, 4, 2, 5, 3)
        up_disp = up_disp.reshape(B, 1, self.scale * H, self.scale * W)

        return up_disp