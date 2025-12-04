from PIL import Image
import numpy as np

def remove_borders_pil(image_path, output_path="clean_pil_result.jpg", tolerance=20):
    """
    Removes dark/black borders from an image using Pillow + NumPy.
    Automatically adapts to dark gray borders.
    
    Args:
        image_path (str): Path to input image.
        output_path (str): Path to save cleaned image.
        tolerance (int): Darkness threshold (lower = stricter, higher = allows darker grays).
    """

    # Step 1: Load image
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)

    # Step 2: Convert to grayscale
    gray = np.mean(arr, axis=2).astype(np.uint8)

    # Step 3: Create a mask of non-dark pixels
    mask = gray > tolerance

    # Step 4: Find bounding box of all non-dark areas
    coords = np.argwhere(mask)
    if coords.size == 0:
        print("⚠️ No non-dark area found. Try lowering tolerance.")
        img.save(output_path)
        return

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # +1 to include the last pixel

    # Step 5: Crop the original image
    cropped = img.crop((x0, y0, x1, y1))

    # Step 6: Save result
    cropped.save(output_path)
    print(f"✅ Border removed and saved as {output_path}")

# Example usage
remove_borders_pil("/Users/shahriarmoradi/Desktop/Work/Roca Technologies/work/dev/development/code_development/dev_github/from_github_mergeconflict_kyc_kyb_13ocbt/output_folder/Document_1 (49).jpg", "clean_pil_output1.jpg", tolerance=25)
