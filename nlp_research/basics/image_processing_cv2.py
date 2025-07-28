import cv2
import numpy as np


def preprocess_receipt_image(image_path: str, output_path: str | None = None) -> np.ndarray:
    """
    Preprocess a receipt image to improve OCR accuracy.
    Args:
        image_path (str): Path to the input image.
        output_path (Optional[str]): If provided, save the processed image to this path.
    Returns:
        np.ndarray: The preprocessed image ready for OCR.
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'Could not read image at {image_path}')

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise (median blur is good for salt-and-pepper noise)
    denoised = cv2.medianBlur(gray, 3)

    # Adaptive thresholding for binarization
    binarized = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15
    )

    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)

    if output_path:
        cv2.imwrite(output_path, cleaned)
    return cleaned


def main():
    image_path = 'receipt.jpg'
    output_path = 'processed_receipt.jpg'
    preprocess_receipt_image(image_path, output_path)


if __name__ == '__main__':
    main()
