import cv2
import numpy as np

def compute_ssim(img1, img2):
    # Ensure the images are float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Constants for SSIM, standard for 8-bit image
    C1 = 6.5025
    C2 = 58.5225

    # Gaussian blur to get local means
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T

    mu1 = cv2.filter2D(img1, -1, window)
    mu2 = cv2.filter2D(img2, -1, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()

def keyframe_extract(path, keyframe_maxcount = -1, min_frame_gap = 240, ssim_threshold = 0.5):
    cap = cv2.VideoCapture(path)
    success, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
    frame_idx = 1
    last_frame_idx = -1
    keyframe_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ssim_score = compute_ssim(prev_gray, gray_frame)

        if ssim_score < ssim_threshold and \
            (frame_idx - last_frame_idx) >= min_frame_gap and \
            is_not_transition_frame(frame) and \
            is_orb_structurally_different(prev_gray, gray_frame) and \
            is_shot_boundary(prev_frame, frame):
            
                cv2.imwrite(f"caps/keyframe_{keyframe_count}_ssim_{ssim_score:.3f}.jpg", frame)
                print(f"Saved keyframe at frame {frame_idx} with SSIM: {ssim_score:.3f}")
                keyframe_count += 1
                
        prev_gray = gray_frame
        prev_frame = frame
        frame_idx += 1

    cap.release()

def is_not_transition_frame(frame, brightness_thresh = 15, color_var_thresh = 10):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    color_std = np.std(frame)
    return mean_brightness > brightness_thresh and color_std > color_var_thresh

def is_orb_structurally_different(img1, img2, match_thresh = 30):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return True

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good_matches) < match_thresh

def is_shot_boundary(prev_frame, curr_frame, hist_thresh=0.5):
    hsv_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    hsv_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)

    hist_prev = cv2.calcHist([hsv_prev], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_curr = cv2.calcHist([hsv_curr], [0, 1], None, [50, 60], [0, 180, 0, 256])

    cv2.normalize(hist_prev, hist_prev)
    cv2.normalize(hist_curr, hist_curr)

    similarity = cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_BHATTACHARYYA)
    return similarity > hist_thresh