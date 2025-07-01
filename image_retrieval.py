import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

IMG_FOLDER = os.getenv("IMG_FOLDER")
IMG_PLOT_DIR = os.getenv("IMG_PLOT_FOLDER")
TARGET_IMAGE = os.getenv("QUERY_IMAGE")

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_histogram(image_path, bins=(50, 60, 60), save_hist_img=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)

    if save_hist_img:
        h_hist = cv2.calcHist([hsv], [0], None, [bins[0]], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [bins[1]], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [bins[2]], [0, 256])

        ensure_dir_exists(os.path.dirname(save_hist_img))

        plt.figure(figsize=(10, 4))
        plt.suptitle(f"Histogram for {os.path.basename(image_path)}", fontsize=14)
        plt.subplot(1, 3, 1)
        plt.plot(h_hist, color='r')
        plt.title('Hue')
        plt.subplot(1, 3, 2)
        plt.plot(s_hist, color='g')
        plt.title('Saturation')
        plt.subplot(1, 3, 3)
        plt.plot(v_hist, color='b')
        plt.title('Value')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(save_hist_img)
        plt.close()

    return hist

def compare_histograms(hist1, hist2):
    h1 = hist1.astype('float32')
    h2 = hist2.astype('float32')
    h1 = h1 / (np.sum(h1) + 1e-10)
    h2 = h2 / (np.sum(h2) + 1e-10)
    chi_sq = cv2.compareHist(h1, h2, cv2.HISTCMP_CHISQR)
    similarity = 1.0 - (chi_sq / (chi_sq + 1))
    return similarity

def extract_sift_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors, img

def save_sift_keypoints(image_path, keypoints, img, save_path):
    ensure_dir_exists(os.path.dirname(save_path))
    img_kp = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"SIFT Keypoints: {os.path.basename(image_path)}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compare_sift_features(desc1, desc2):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    if not matches:
        return 0.0
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 200]
    similarity = len(good_matches) / max(len(desc1), len(desc2))
    return min(similarity, 1.0)

if __name__ == "__main__":
    result_rows_hist = []
    result_rows_sift = []

    ensure_dir_exists(IMG_PLOT_DIR)

    files = [f for f in os.listdir(IMG_FOLDER) if os.path.isfile(os.path.join(IMG_FOLDER, f))]
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    target_hist = extract_histogram(TARGET_IMAGE, save_hist_img=f"{IMG_PLOT_DIR}/target_histogram.png")
    target_kp, target_desc, target_img = extract_sift_features(TARGET_IMAGE)
    save_sift_keypoints(TARGET_IMAGE, target_kp, target_img, f"{IMG_PLOT_DIR}/target_sift.png")

    for file_name in files:
        file_path = os.path.join(IMG_FOLDER, file_name)
        hist = extract_histogram(file_path, save_hist_img=f"{IMG_PLOT_DIR}/{os.path.splitext(file_name)[0]}_histogram.png")
        color_similarity = compare_histograms(target_hist, hist)
        result_rows_hist.append({
            "File": file_name,
            "Color_similarity": round(color_similarity, 3)
        })
        print(f"Comparing {TARGET_IMAGE} with {file_path}: Color_similarity: {color_similarity:.3f}")

    df_hist = pd.DataFrame(result_rows_hist).sort_values(by="Color_similarity", ascending=False).reset_index(drop=True)
    df_hist.index += 1

    for file_name in files:
        file_path = os.path.join(IMG_FOLDER, file_name)
        kp, desc, img = extract_sift_features(file_path)
        save_sift_keypoints(file_path, kp, img, f"{IMG_PLOT_DIR}/{os.path.splitext(file_name)[0]}_sift.png")
        sift_similarity = compare_sift_features(target_desc, desc)
        result_rows_sift.append({
            "File": file_name,
            "SIFT_similarity": round(sift_similarity, 3)
        })
        print(f"Comparing {TARGET_IMAGE} with {file_path}: SIFT_similarity: {sift_similarity:.3f}")

    df_sift = pd.DataFrame(result_rows_sift).sort_values(by="SIFT_similarity", ascending=False).reset_index(drop=True)
    df_sift.index += 1

    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.5 * len(df_hist)))
    ax.axis('off')
    table = ax.table(cellText=df_hist.values, colLabels=df_hist.columns, rowLabels=df_hist.index,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df_hist.columns) + 1)))
    plt.title(f"Image Color Histogram Similarity Ranking (to {TARGET_IMAGE})", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("result_hist.png")
    plt.close()
    print("\nColor histogram ranking table saved as result_hist.png")

    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.5 * len(df_sift)))
    ax.axis('off')
    table = ax.table(cellText=df_sift.values, colLabels=df_sift.columns, rowLabels=df_sift.index,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df_sift.columns) + 1)))
    plt.title(f"Image SIFT Similarity Ranking (to {TARGET_IMAGE})", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("result_sift.png")
    plt.close()
    print("\nSIFT ranking table saved as result_sift.png")
