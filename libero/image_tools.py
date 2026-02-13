import numpy as np
from PIL import Image

def pil_image_to_256_matrix(path: str) -> np.ndarray:
    """
    Load image, convert to RGB, resize to 256x256, and return as numpy array.
    
    Returns:
        np.ndarray: Image array with shape (256, 256, 3) and dtype uint8
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256), resample=Image.LANCZOS)
    x = np.array(img, dtype=np.uint8)  
    return x


def hwc_rgb_to_pil(x: np.ndarray) -> Image.Image:
    """
    Convert HWC RGB numpy array to PIL Image.
    
    Args:
        x: Numpy array with shape (H, W, 3), e.g., (256, 256, 3)
           dtype can be uint8, float32, float64, etc.
    
    Returns:
        PIL.Image: RGB image in PIL format
    """
    x = np.asarray(x)
    assert x.ndim == 3 and x.shape[2] == 3, f"Expected shape (H, W, 3), got {x.shape}"

    if np.issubdtype(x.dtype, np.floating):
        mx = float(np.nanmax(x))
        if mx <= 1.0 + 1e-6:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    else:
        x = np.clip(x, 0, 255).astype(np.uint8)

    return Image.fromarray(x, mode="RGB")


def process_libero_observation(obs: dict) -> tuple:
    """
    Process LIBERO environment observations with 180° rotation and PIL conversion.
    
    Args:
        obs: Observation dictionary returned by LIBERO environment
        
    Returns:
        tuple: (agentview_pil, wrist_pil, agentview_array) where:
            - agentview_pil: Third-person view PIL image
            - wrist_pil: Wrist camera PIL image
            - agentview_array: Third-person view numpy array for video saving
    """
    # Rotate 180° to match training data preprocessing
    agentview_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    
    # Convert to PIL images
    agentview_pil = hwc_rgb_to_pil(agentview_img)
    wrist_pil = hwc_rgb_to_pil(wrist_img)
    
    # Return numpy array for video saving
    agentview_array = np.array(agentview_pil)
    
    return agentview_pil, wrist_pil, agentview_array