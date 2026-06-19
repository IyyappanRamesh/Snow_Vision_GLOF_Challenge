import os
import glob
import cv2
import numpy as np
import torch
from tqdm import tqdm
# Import architecture layout directly from standard local tracking definition
from model_architecture import TransUNetHybrid

def execute_batch_inference():
    # --- CHALLENGE REQUIRED MATRIX PATH CONFIGURATIONS ---
    INPUT_FOLDER_NAME = "GLOFeagles 2026 challenge dataset"   #give dataset path here
    OUTPUT_FOLDER_NAME = "GLOFeagles_2026_segmented_masks"    #give output folder destiny here
    WEIGHTS_FILE_NAME = "best_transunet_model.pt"              #Give the path of provided best fine tuned model .pt file here

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Structural compilation mapping out model setup
    model = TransUNetHybrid()
    
    if os.path.exists(WEIGHTS_FILE_NAME):
        model.load_state_dict(torch.load(WEIGHTS_FILE_NAME, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Core checkpoint parameters loaded successfully from: {WEIGHTS_FILE_NAME}")
    else:
        print(f"Aborted: Weights file '{WEIGHTS_FILE_NAME}' missing from local working root.")
        return

    os.makedirs(OUTPUT_FOLDER_NAME, exist_ok=True)
    
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.PNG', '*.JPG')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER_NAME, ext)))

    print(f"Evaluating challenge pool: Total of {len(image_paths)} image frames discovered.")

    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Segmenting Images"):
            filename = os.path.basename(img_path)
            raw_bgr = cv2.imread(img_path)
            if raw_bgr is None: continue
            
            orig_h, orig_w, _ = raw_bgr.shape
            
            # Formatting input spatial configuration bounds
            rgb_img = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(rgb_img, (512, 512), interpolation=cv2.INTER_LINEAR)
            normalized_arr = resized_img.astype(np.float32) / 255.0
            input_tensor = torch.from_numpy(normalized_arr).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Predict
            output_logits = model(input_tensor)
            prob_mask = torch.sigmoid(output_logits).squeeze().cpu().numpy()
            
            # Mathematical evaluation mask assignment
            binary_mask = (prob_mask >= 0.5).astype(np.uint8) * 255
            final_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # Write out mask using unified submission labeling pattern
            output_filepath = os.path.join(OUTPUT_FOLDER_NAME, f"mask_{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(output_filepath, final_mask)

    print(f"Batch run complete. Segmentations exported cleanly to: {OUTPUT_FOLDER_NAME}/")

if __name__ == "__main__":
    if os.path.exists("GLOFeagles 2026 challenge dataset"):
        execute_batch_inference()
    else:
        print("Module compiled cleanly. Ready to run when challenge directories are unzipped.")
