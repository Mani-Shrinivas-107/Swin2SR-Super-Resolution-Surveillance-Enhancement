import os
import cv2
import torch
import numpy as np
from models.network_swins2sr import Swin2SR 

def run_super_resolution(img_path, model, device, output_path):
    # Load & Preprocess Image
    img_lq = cv2.imread(img_path) / 255.
    if img_lq is None: return # Skip if image is corrupt
    img_lq = torch.from_numpy(np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_lq = img_lq.unsqueeze(0).to(device)

    # Tiled Inference Settings
    tile_size = 128  
    tile_overlap = 16 
    batch_size, channels, height, width = img_lq.size()
    stride = tile_size - tile_overlap
    
    output_shape = (batch_size, channels, height * 4, width * 4)
    output = torch.zeros(output_shape).to(device)
    output_mask = torch.zeros(output_shape).to(device)

    print(f"Processing: {os.path.basename(img_path)}")
    
    with torch.no_grad():
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # ACTUAL PROGRESS PRINTING
                print(f"  > Progress: Row {y//stride + 1}, Col {x//stride + 1}", end='\r')
                
                # Extract tile
                y2 = min(y + tile_size, height)
                x2 = min(x + tile_size, width)
                y1 = max(0, y2 - tile_size)
                x1 = max(0, x2 - tile_size)
                
                tile = img_lq[:, :, y1:y2, x1:x2]
                
                # Pad tile to be multiple of 8
                h_pad = (8 - tile.shape[2] % 8) % 8
                w_pad = (8 - tile.shape[3] % 8) % 8
                tile = torch.nn.functional.pad(tile, (0, w_pad, 0, h_pad), mode='reflect')

                # Run model
                tile_output, _ = model(tile)
                
                # Unpad and accumulate
                tile_output = tile_output[:, :, : (y2-y1)*4, : (x2-x1)*4]
                output[:, :, y1*4:y2*4, x1*4:x2*4] += tile_output
                output_mask[:, :, y1*4:y2*4, x1*4:x2*4] += 1.0

        output /= output_mask # Average overlaps

    # Save Result
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    cv2.imwrite(output_path, output)
    print(f"\nDone! Saved to {output_path}")

# --- SETUP ---
base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, 'inputs')
weights_file = os.path.join(base_dir, 'weights', 'Swin2SR_CompressedSR_X4_48.pth')
results_dir = os.path.join(base_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model ONCE outside the loop
print("Loading Model & Weights...")
model = Swin2SR(upscale=4, in_chans=3, img_size=48, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], 
                embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], 
                mlp_ratio=2, upsampler='pixelshuffle_aux', resi_connection='1conv')

pretrained_model = torch.load(weights_file, map_location=device)
model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model else pretrained_model, strict=True)
model.eval().to(device)
print("Model ready.")

# Process all images
if os.path.exists(input_dir):
    for img_file in os.listdir(input_dir):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            run_super_resolution(
                os.path.join(input_dir, img_file),
                model, device,
                os.path.join(results_dir, f"enhanced_{img_file}")
            )