import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.signal import cwt, ricker
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def apply_spatial_filter(image, method='highpass'):
    """Applying spatial filtering to an image using a high-pass filter."""
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) if method == 'highpass' else None
    return cv2.filter2D(image, -1, kernel)

def histogram_equalization(image):
    """Applying histogram equalization to enhance image contrast."""
    return cv2.equalizeHist(image)

def frequency_domain_filtering(image):
    """Filtering the image in the frequency domain to obtain the magnitude spectrum."""
    f = fft2(image)
    fshift = fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # adding 1 to avoid log(0)
    return magnitude_spectrum

def compute_continuous_wavelet_transform(image, scales):
    """Computing the Continuous Wavelet Transform (CWT) of an image."""
    # In a real-world scenario, we would work with a 1D signal.
    # For simplicity, collapsing the image into a 1D signal by averaging across one dimension.
    signal = np.mean(image, axis=0)
    cwt_result = cwt(signal, ricker, scales)
    return cwt_result

def visualize_cwt(cwt_result, scales, title="Continuous Wavelet Transform"):
    """Visualizing the Continuous Wavelet Transform."""
    plt.imshow(np.abs(cwt_result), extent=[0, 1, scales[0], scales[-1]], cmap='viridis', aspect='auto')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency (Scale)')
    plt.show()

def detect_keypoints_with_sift(image):
    """Detecting keypoints in an image using the SIFT detector."""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image_with_keypoints, keypoints

def visualize_vibrations(image, title="Vibration Analysis"):
    """Visualizing the processed image."""
    plt.imshow(image, cmap = 'gray')
    plt.title(title)
    plt.show()

def calculate_metrics(original, processed):
    """Calculating SSIM, PSNR, and MSE metrics."""
    ssim_value = ssim(original, processed)
    psnr_value = psnr(original, processed)
    mse_value = mse(original, processed)
    return ssim_value, psnr_value, mse_value

# Example usage:
image_path = "test_image.png"
original_image = load_image(image_path)

# Applying spatial filtering and histogram equalization:
filtered_image = apply_spatial_filter(original_image)
equalized_image = histogram_equalization(filtered_image)

# Frequency domain analysis:
magnitude_spectrum = frequency_domain_filtering(equalized_image)
visualize_vibrations(magnitude_spectrum, "Magnitude Spectrum")

# CWT analysis:
scales = np.arange(1, 31)
cwt_result = compute_continuous_wavelet_transform(equalized_image, scales)
visualize_cwt(cwt_result, scales)

# Hessian-based point of interest detection:
image_with_keypoints, keypoints = detect_keypoints_with_sift(equalized_image)
visualize_vibrations(image_with_keypoints, "Points of Interest")

###########################################################################################

def compare_images(reference_image, test_image):
    ssim_value = ssim(reference_image, test_image)
    psnr_value = psnr(reference_image, test_image)
    mse_value = mse(reference_image, test_image)
    return ssim_value, psnr_value, mse_value

def is_vibration_excessive(ssim_value, psnr_value, mse_value, ssim_threshold, psnr_threshold, mse_threshold):
    """Check if the vibration is excessive based on the thresholds."""
    if ssim_value < ssim_threshold or psnr_value < psnr_threshold or mse_value > mse_threshold:
        return True  # Vibration is excessive
    else:
        return False  # Vibration is within normal range

def resize_to_match(image1, image2):
    """
    Resize the first image to match the dimensions of the second image.
    """
    height, width = image2.shape[:2]
    resized_image = cv2.resize(image1, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image

# Load the reference image and process it
reference_image_path = "reference_image.png"
reference_image = load_image(reference_image_path)
processed_reference_image = histogram_equalization(apply_spatial_filter(reference_image))

# Load the test image and process it
test_image_path = "test_image.png"
test_image = load_image(test_image_path)
processed_test_image = histogram_equalization(apply_spatial_filter(test_image))

# Resize the reference image to match the test image dimensions
resized_reference_image = resize_to_match(processed_reference_image, processed_test_image)

# Now that both images are of the same size, you can compare them
ssim_val, psnr_val, mse_val = compare_images(resized_reference_image, processed_test_image)

# Define your thresholds here based on your empirical data
ssim_threshold = 0.85
psnr_threshold = 30
mse_threshold = 500

# Check if vibration is excessive
excessive_vibration = is_vibration_excessive(ssim_val, psnr_val, mse_val, ssim_threshold, psnr_threshold, mse_threshold)

# Report the results
print(f"SSIM: {ssim_val}, PSNR: {psnr_val}, MSE: {mse_val}")
print("Excessive Vibration Detected!" if excessive_vibration else "Vibration within normal range.")
