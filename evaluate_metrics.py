import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

def evaluate_folder(generated_folder, ground_truth_folder, output_csv="evaluation_results.csv"):
    results = []
    print(f"Evaluating Generated Images in: {generated_folder}")
    print(f"Against Ground Truth in: {ground_truth_folder}\n")
    
    # Recursively find all generated images
    gen_images = []
    for root, _, files in os.walk(generated_folder):
        for file in files:
            if file.endswith(("_fake_B.png", "_fake_B.jpg", ".png", ".jpg")):
                gen_images.append(os.path.join(root, file))

    if len(gen_images) == 0:
        print("No generated images found! Ensure you run 'test.py' first to generate inference images.")
        return

    # To search for ground truth dynamically despite subfolder structures
    gt_images = {}
    print("Indexing Ground Truth Cityscapes Dataset...")
    for root, _, files in os.walk(ground_truth_folder):
        for file in files:
            if file.endswith(('.png', '.jpg')):
                # Normalize cityscapes names by removing domain specific suffixes to allow matching
                normalized_name = file.replace("_leftImg8bit_foggy", "_leftImg8bit")
                gt_images[normalized_name] = os.path.join(root, file)

    psnr_list = []
    ssim_list = []

    print("Calculating PSNR and SSIM Metrics...")
    for gen_path in gen_images:
        filename = os.path.basename(gen_path)
        
        # Recover baseline filename from CycleGAN output
        base_name = filename.replace("_fake_B", "").replace("_foggy", "")
        
        # Try to find matching GT
        gt_path = None
        for key in gt_images.keys():
            if key in base_name or base_name in key:
                gt_path = gt_images[key]
                break

        if gt_path is None:
            continue

        gen_img = cv2.imread(gen_path)
        gt_img = cv2.imread(gt_path)

        if gen_img is None or gt_img is None:
            continue

        # Resize GT to match Generated if needed (CycleGAN defaults to 256x256)
        if gen_img.shape != gt_img.shape:
            gt_img = cv2.resize(gt_img, (gen_img.shape[1], gen_img.shape[0]))

        # Calculate metrics
        ssim_val = compute_ssim(gt_img, gen_img, channel_axis=2)
        psnr_val = compute_psnr(gt_img, gen_img)
        
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        
        results.append({
            "Image": filename,
            "PSNR": psnr_val,
            "SSIM": ssim_val
        })

    if not results:
        print("\nCould not find matching ground truth 'Clear' images for the generated 'Foggy' outputs.")
        print("Note: If the datasets are entirely un-paired (i.e., no overlapping street scenes between Domain A and B), PSNR/SSIM cannot be calculated pixel-by-pixel. Visual qualitative assessment is recommended instead.")
        return

    df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_csv)
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("      ✨ FINAL EVALUATION METRICS ✨      ")
    print("="*50)
    print(f"Total Images Evaluated: {len(results)}")
    print(f"Average PSNR: {np.mean(psnr_list):.4f} dB (Higher is better)")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}      (1.0 is mathematically identical)")
    print(f"Detailed results saved to: {output_path}")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_dir', type=str, required=True, help="Path to CycleGAN test results folder")
    parser.add_argument('--groundtruth_dir', type=str, required=True, help="Path to Real Clear Ground Truth images")
    args = parser.parse_args()
    
    evaluate_folder(args.generated_dir, args.groundtruth_dir)
