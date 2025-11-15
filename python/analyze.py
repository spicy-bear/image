#!/usr/bin/env python3
"""
SAR Interactive Color Detector with Auto-Analysis
Runs full color analysis on upload, then allows a full-video
batch scan for ALL detection methods (HSV, Sat, YCrCb, ML).
"""

from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
import io
from pathlib import Path
import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist # Import cdist for ML detection

app = Flask(__name__)

# Global state
video_path = None
frame_cache = {}
total_frames = 0
fps = 30
analysis_results = None
ml_results_cache = None
ml_outlier_pixels = None  # Store which pixels are ML outliers

def analyze_video_colors(video_path, sample_rate=10, subsample=20, min_saturation=120):
    """
    Analyze video colors and return statistics
    Same as the smart analysis script
    """
    print(f"\n🎨 Running color analysis (sampling every {sample_rate} frame(s), pixel subsample: {subsample})...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_hues = Counter()
    vivid_hues = Counter()
    saturation_values = []
    
    total_pixels = 0
    vivid_pixels = 0
    frame_count = 0
    frames_sampled = 0
    
    frame_indices = list(range(0, total_frames, sample_rate))
    
    for frame_num in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frames_sampled += 1
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        
        # Use the new subsample parameter here
        for y in range(0, h, subsample):
            for x in range(0, w, subsample):
                hue = hsv[y, x, 0]
                sat = hsv[y, x, 1]
                val = hsv[y, x, 2]
                
                if 20 < val < 240:
                    all_hues[hue] += 1
                    saturation_values.append(sat)
                    total_pixels += 1
                    
                    if sat >= min_saturation:
                        vivid_hues[hue] += 1
                        vivid_pixels += 1
        
        if frames_sampled % 20 == 0:
            print(f"   Analyzed {frames_sampled}/{len(frame_indices)} frames...")
    
    cap.release()
    
    if total_pixels == 0:
        total_pixels = 1 # Avoid division by zero
        
    print(f"✅ Analysis complete: {total_pixels:,} pixels, {vivid_pixels:,} vivid ({vivid_pixels/total_pixels*100:.1f}%)")
    
    # Calculate statistics
    sat_array = np.array(saturation_values)
    
    # Group vivid colors by family
    color_groups = {
        'Red (0-15)': sum(vivid_hues[h] for h in range(0, 16)),
        'Orange (16-30)': sum(vivid_hues[h] for h in range(16, 31)),
        'Yellow (31-45)': sum(vivid_hues[h] for h in range(31, 46)),
        'Green (46-85)': sum(vivid_hues[h] for h in range(46, 86)),
        'Cyan (86-115)': sum(vivid_hues[h] for h in range(86, 116)),
        'Blue (116-145)': sum(vivid_hues[h] for h in range(116, 146)),
        'Purple (146-165)': sum(vivid_hues[h] for h in range(146, 166)),
        'Magenta (166-179)': sum(vivid_hues[h] for h in range(166, 180)),
    }
    
    # Top vivid hues
    top_vivid = vivid_hues.most_common(10)
    
    results = {
        'total_pixels': total_pixels,
        'vivid_pixels': vivid_pixels,
        'vivid_percent': vivid_pixels / total_pixels * 100,
        'sat_mean': float(sat_array.mean()) if len(sat_array) > 0 else 0,
        'sat_median': float(np.median(sat_array)) if len(sat_array) > 0 else 0,
        'sat_p25': float(np.percentile(sat_array, 25)) if len(sat_array) > 0 else 0,
        'sat_p75': float(np.percentile(sat_array, 75)) if len(sat_array) > 0 else 0,
        'color_groups': color_groups,
        'top_vivid_hues': [(int(h), int(c)) for h, c in top_vivid],
        'frames_analyzed': frames_sampled
    }
    
    return results

def get_frame(frame_num):
    """Get a frame from video with caching"""
    global video_path, frame_cache
    
    if video_path is None:
        return None
    
    if frame_num in frame_cache:
        return frame_cache[frame_num]
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Limit cache size
    if len(frame_cache) > 50:
        # Remove the first (oldest) item
        frame_cache.pop(next(iter(frame_cache)))
    
    frame_cache[frame_num] = frame.copy()
    return frame

def encode_image(frame, quality=85):
    """Encode an image as a base64 string"""
    if frame is None:
        return ""
    try:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('utf-8')
    except cv2.error:
        # Handle cases where frame might be empty or invalid
        return ""

def get_color_name(hue):
    """Get color name from hue"""
    if hue <= 15 or hue >= 170:
        return "Red"
    elif hue <= 30:
        return "Orange"
    elif hue <= 45:
        return "Yellow"
    elif hue <= 85:
        return "Green"
    elif hue <= 115:
        return "Cyan"
    elif hue <= 145:
        return "Blue"
    elif hue <= 165:
        return "Purple"
    else:
        return "Magenta"

# --- MOVED TO GLOBAL SCOPE ---
def draw_detections(frame, contours, label_prefix, label_data_func, color, thickness=2):
    """
    Helper function to draw detection boxes and labels on a frame.
    label_data_func takes (x, y, w, h) and returns a string.
    """
    frame_out = frame.copy()
    detections = []
    detection_count = 0
    
    for contour in contours:
        detection_count += 1
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Get data for label
        data_string = label_data_func(x, y, w, h)
        
        detections.append({
            'x': x, 'y': y, 'w': w, 'h': h,
            'area': int(area),
            'label': f"{label_prefix}: {data_string}"
        })
        
        # Draw box
        cv2.rectangle(frame_out, (x, y), (x+w, y+h), color, thickness)
        
        # Draw label
        label = f"{label_prefix}: {data_string}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        
        cv2.rectangle(frame_out, (x, y - label_size[1] - 8),
                      (x + label_size[0] + 8, y), color, -1)
        cv2.putText(frame_out, label, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    return frame_out, detections, detection_count

def filter_contours(mask, min_size, max_size):
    """
    Helper function to filter contours from a mask by area.
    Returns the list of filtered contours and a new filtered mask.
    """
    if mask is None or np.sum(mask) == 0:
        return [], np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_size <= area <= max_size:
            filtered_contours.append(c)
    
    filtered_mask = np.zeros_like(mask)
    if filtered_contours:
        cv2.drawContours(filtered_mask, filtered_contours, -1, (255), -1)
    return filtered_contours, filtered_mask
# --- END GLOBAL SCOPE MOVE ---


def detect_ml_outliers_in_frame(frame, method='dbscan', terrain_std_dev_mult=5.0):
    """
    Detect ML outliers in a single frame using trained ML models.
    Uses SAMPLING (step=10) to avoid memory errors.
    """
    global ml_results_cache
    h, w = frame.shape[:2]

    if ml_results_cache is None:
        return np.zeros((h, w), dtype=np.uint8), np.full((h, w), -2, dtype=int)

    scaler = ml_results_cache.get('scaler')
    pca = ml_results_cache.get('pca')
    kmeans = ml_results_cache.get('kmeans')
    terrain_centroid = ml_results_cache.get('terrain_centroid')
    
    # --- NEW: Calculate radius dynamically based on param ---
    terrain_radius = 10.0 # Default fallback
    terrain_std = ml_results_cache.get('terrain_std')
    if terrain_std is not None and len(terrain_std) > 0:
        terrain_radius = np.mean(terrain_std) * terrain_std_dev_mult
    # --- END NEW ---
        
    outlier_sat_threshold = ml_results_cache.get('outlier_sat_threshold', 100)

    if scaler is None or pca is None:
        return np.zeros((h, w), dtype=np.uint8), np.full((h, w), -2, dtype=int)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    pixels_for_ml = []
    pixel_coords = []
    step = 10 # Sample every 10x10 pixels
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            hue = hsv[y, x, 0]
            sat = hsv[y, x, 1]
            val = hsv[y, x, 2]
            l = lab[y, x, 0]
            a = lab[y, x, 1]
            b = lab[y, x, 2]

            if sat > 15 and 20 < val < 240:  # Lower threshold for detection
                pixels_for_ml.append([hue, sat, val, l, a, b])
                pixel_coords.append((y, x))
    
    if len(pixels_for_ml) == 0:
        return np.zeros((h, w), dtype=np.uint8), np.full((h, w), -2, dtype=int)

    pixels_for_ml = np.array(pixels_for_ml)
    pixels_scaled = scaler.transform(pixels_for_ml)
    pixels_pca = pca.transform(pixels_scaled)

    full_frame_labels = np.full((h, w), -2, dtype=int) # -2 for non-processed
    detection_mask = np.zeros((h, w), dtype=np.uint8)
    
    if method == 'dbscan':
        if terrain_centroid is not None and terrain_radius is not None:
            # TWO-CONDITION DETECTION: distance from terrain + high saturation
            distances_from_terrain = np.linalg.norm(pixels_pca - terrain_centroid, axis=1)
            is_far_from_terrain = distances_from_terrain > terrain_radius
            
            # Check saturation
            saturations = pixels_for_ml[:, 1]
            has_high_saturation = saturations > outlier_sat_threshold
            
            # BOTH must be true
            labels = np.where(is_far_from_terrain & has_high_saturation, -1, 0)
        else:
            labels = np.zeros(len(pixels_pca)) # No model, no labels
    elif method == 'pca_outliers':
        distances_from_mean = np.linalg.norm(pixels_pca - np.mean(pixels_pca, axis=0), axis=1)
        threshold = np.mean(distances_from_mean) + 5 * np.std(distances_from_mean)  # EXTREME
        
        saturations = pixels_for_ml[:, 1]
        has_high_sat = saturations > outlier_sat_threshold
        
        labels = np.where((distances_from_mean > threshold) & has_high_sat, -1, 0)
    else:
        # K-means or other
        if kmeans is not None:
             labels = kmeans.predict(pixels_pca)
        else:
             labels = np.zeros(len(pixels_pca)) # Default to no labels

    # Reconstruct by drawing blocks
    block_size = step
    for i, (y, x) in enumerate(pixel_coords):
        label_val = int(labels[i]) # Ensure it's an integer
        # Define a block
        y_min, y_max = max(0, y - block_size // 2), min(h, y + block_size // 2 + 1)
        x_min, x_max = max(0, x - block_size // 2), min(w, x + block_size // 2 + 1)
        
        # Fill the block with the label
        full_frame_labels[y_min:y_max, x_min:x_max] = label_val
        
        # Update the detection mask
        if label_val == -1: # PCA or DBSCAN outlier
            detection_mask[y_min:y_max, x_min:x_max] = 255
        elif method == 'kmeans' and ml_results_cache.get('outlier_kmeans_cluster_label') is not None and \
             label_val == ml_results_cache['outlier_kmeans_cluster_label']:
            detection_mask[y_min:y_max, x_min:x_max] = 255

    return detection_mask, full_frame_labels

# --- NEW BATCH PROCESSING HELPER ---
def process_frame_for_batch(frame, params):
    """
    Processes a single frame for all detection methods based on params.
    Returns a frame with boxes drawn and a list of detections.
    """
    
    # --- 1. Get Channels ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hue_channel, sat_channel, val_channel = cv2.split(hsv)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    
    all_contours_to_draw = []
    all_detections_for_list = []
    
    morph_size = params['morph_size']
    kernel = None
    if morph_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))

    # --- 2. HSV Detections ---
    if params['hsv_hue_min'] <= params['hsv_hue_max']:
        hue_mask = (hue_channel >= params['hsv_hue_min']) & (hue_channel <= params['hsv_hue_max'])
    else: # Wrap around
        hue_mask = (hue_channel >= params['hsv_hue_min']) | (hue_channel <= params['hsv_hue_max'])
    
    sat_mask = (sat_channel >= params['hsv_sat_min']) & (sat_channel <= params['hsv_sat_max'])
    val_mask = (val_channel >= params['hsv_val_min']) & (val_channel <= params['hsv_val_max'])
    
    exclude_mask = np.zeros(hue_channel.shape, dtype=bool)
    for color in params['excluded_colors']:
        if color == 'Red':
            exclude_mask |= ((hue_channel >= 0) & (hue_channel <= 15)) | ((hue_channel >= 170) & (hue_channel <= 179))
        elif color == 'Orange':
            exclude_mask |= (hue_channel >= 16) & (hue_channel <= 30)
        # ... (add other colors if needed, skipping for brevity)
            
    hsv_detection_mask = (hue_mask & sat_mask & val_mask & ~exclude_mask).astype(np.uint8) * 255

    if params['exclude_sky']:
        sky_mask = ((hue_channel >= 90) & (hue_channel <= 120) & (val_channel > 180) & (sat_channel < 100))
        hsv_detection_mask = cv2.bitwise_and(hsv_detection_mask, cv2.bitwise_not(sky_mask.astype(np.uint8) * 255))
    
    if kernel is not None:
        hsv_detection_mask = cv2.morphologyEx(hsv_detection_mask, cv2.MORPH_OPEN, kernel)
        hsv_detection_mask = cv2.morphologyEx(hsv_detection_mask, cv2.MORPH_CLOSE, kernel)
        
    hsv_contours, hsv_mask_filtered = filter_contours(hsv_detection_mask, params['hsv_min_size'], params['hsv_max_size'])
    if len(hsv_contours) > 0:
        def hsv_label_func(x, y, w, h):
            roi_hue = hue_channel[y:y+h, x:x+w]
            roi_sat = sat_channel[y:y+h, x:x+w]
            roi_mask_pixels = hsv_mask_filtered[y:y+h, x:x+w] > 0
            if np.sum(roi_mask_pixels) == 0: return "N/A"
            avg_hue = int(np.mean(roi_hue[roi_mask_pixels]))
            avg_sat = int(np.mean(roi_sat[roi_mask_pixels]))
            return f"{get_color_name(avg_hue)} H:{avg_hue} S:{avg_sat}"
        all_contours_to_draw.append((hsv_contours, "HSV", hsv_label_func, (0, 255, 0))) # Green

    # --- 3. Saturation Detections ---
    sat_detection_mask = (sat_channel >= params['sat_sat_min']).astype(np.uint8) * 255
    if kernel is not None:
        sat_detection_mask = cv2.morphologyEx(sat_detection_mask, cv2.MORPH_OPEN, kernel)
    
    sat_contours, sat_mask_filtered = filter_contours(sat_detection_mask, params['sat_min_size'], params['sat_max_size'])
    if len(sat_contours) > 0:
        def sat_label_func(x, y, w, h):
            roi_sat = sat_channel[y:y+h, x:x+w]
            roi_mask_pixels = sat_mask_filtered[y:y+h, x:x+w] > 0
            if np.sum(roi_mask_pixels) == 0: return "N/A"
            avg_sat = int(np.mean(roi_sat[roi_mask_pixels]))
            return f"S:{avg_sat}"
        all_contours_to_draw.append((sat_contours, "SAT", sat_label_func, (255, 0, 255))) # Magenta
        
    # --- 4. YCrCb Detections ---
    cr_mask = (cr_channel >= params['ycrcb_cr_min']) & (cr_channel <= params['ycrcb_cr_max'])
    cb_mask = (cb_channel >= params['ycrcb_cb_min']) & (cb_channel <= params['ycrcb_cb_max'])
    ycrcb_detection_mask = (cr_mask & cb_mask).astype(np.uint8) * 255
    if kernel is not None:
        ycrcb_detection_mask = cv2.morphologyEx(ycrcb_detection_mask, cv2.MORPH_OPEN, kernel)
        
    ycrcb_contours, ycrcb_mask_filtered = filter_contours(ycrcb_detection_mask, params['ycrcb_min_size'], params['ycrcb_max_size'])
    if len(ycrcb_contours) > 0:
        def ycrcb_label_func(x, y, w, h):
            roi_cr = cr_channel[y:y+h, x:x+w]
            roi_cb = cb_channel[y:y+h, x:x+w]
            roi_mask_pixels = ycrcb_mask_filtered[y:y+h, x:x+w] > 0
            if np.sum(roi_mask_pixels) == 0: return "N/A"
            avg_cr = int(np.mean(roi_cr[roi_mask_pixels]))
            avg_cb = int(np.mean(roi_cb[roi_mask_pixels]))
            return f"Cr:{avg_cr} Cb:{avg_cb}"
        all_contours_to_draw.append((ycrcb_contours, "YCrCb", ycrcb_label_func, (0, 255, 255))) # Yellow

    # --- 5. ML Detections (DBSCAN) ---
    ml_mask, _ = detect_ml_outliers_in_frame(frame, 'dbscan', params['ml_terrain_stdev'])
    if kernel is not None:
        ml_mask = cv2.morphologyEx(ml_mask, cv2.MORPH_OPEN, kernel)
        ml_mask = cv2.morphologyEx(ml_mask, cv2.MORPH_CLOSE, kernel)
    
    ml_contours, _ = filter_contours(ml_mask, params['ml_min_size'], params['ml_max_size'])
    if len(ml_contours) > 0:
        def ml_label_func(x, y, w, h):
            return "Outlier"
        all_contours_to_draw.append((ml_contours, "ML", ml_label_func, (255, 255, 0))) # Cyan

    # --- 6. Draw results if any ---
    if len(all_contours_to_draw) > 0:
        frame_out = frame.copy()
        for (contours, prefix, lab_func, color) in all_contours_to_draw:
            frame_out, dets, _ = draw_detections(frame_out, contours, prefix, lab_func, color)
            all_detections_for_list.extend(dets)
        return frame_out, all_detections_for_list
    
    # No detections found
    return None, []
# --- END BATCH HELPER ---


@app.route('/')
def index():
    """Serve the main page"""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>SAR Interactive Detector</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #1a1a1a;
            color: #fff;
        }
        .container {
            max-width: 100%;
            margin: 0 auto;
            padding: 0 20px;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        h2 {
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        h3 {
            color: #ddd;
            border-bottom: 1px solid #555;
            padding-bottom: 5px;
        }

        /* Analysis Results Panel */
        #analysis-panel, #ml-analysis-panel {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 2px solid #4CAF50;
        }
        #ml-analysis-panel {
            border-color: #FF9800;
        }
        .analysis-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .stat-box {
            background: #333;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        #ml-analysis-panel .stat-value {
            color: #FF9800;
        }
        .stat-label {
            font-size: 12px;
            color: #aaa;
            margin-top: 5px;
        }
        .color-groups {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .color-group-item {
            background: #333;
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
        }
        
        /* Controls */
        .controls-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .controls {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .control-group {
            margin-bottom: 15px;
            display: grid;
            grid-template-columns: 100px 1fr 60px;
            gap: 10px;
            align-items: center;
        }
         .control-group.dual-slider {
            grid-template-columns: 100px 1fr 1fr 80px;
        }
        .control-group label {
            font-weight: bold;
            font-size: 13px;
        }
        input[type="range"] {
            width: 100%;
            margin: 0;
        }
        input[type="number"] {
            background: #333;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 5px;
            width: 60px;
        }
        .control-group span {
            font-size: 12px;
            white-space: nowrap;
        }
        
        .upload-controls {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .preset-buttons {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        button, .preset-btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 15px;
            cursor: pointer;
            border-radius: 4px;
            font-size: 13px;
        }
        button:hover, .preset-btn:hover {
            background: #45a049;
        }
        .preset-btn {
            background: #2196F3;
        }
        .preset-btn:hover {
            background: #0b7dda;
        }
        
        /* Modal Lightbox */
        .modal {
            display: none;
            position: fixed;
            z-index: 1001;
            padding-top: 50px;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.95);
        }
        .modal-content {
            margin: auto;
            display: block;
            width: 90%;
            max-width: 1600px;
            height: 90vh;
            object-fit: contain;
        }
        .modal .close {
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            transition: 0.3s;
            cursor: pointer;
        }
        
        /* Modal Navigation */
        .modal-prev, .modal-next {
            cursor: pointer;
            position: absolute;
            top: 50%;
            width: auto;
            padding: 16px;
            margin-top: -50px;
            color: white;
            font-weight: bold;
            font-size: 30px;
            transition: 0.3s;
            border-radius: 0 3px 3px 0;
            user-select: none;
            -webkit-user-select: none;
            background-color: rgba(0,0,0,0.3);
        }
        .modal-next {
            right: 0;
            border-radius: 3px 0 0 3px;
        }
        .modal-prev {
            left: 0;
        }
        .modal-prev:hover, .modal-next:hover {
            background-color: rgba(0,0,0,0.6);
        }
        
        /* Loading Overlay */
        #loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        #loading-text {
            margin-top: 20px;
            font-size: 18px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4CAF50;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Exclude Colors */
        .exclude-colors {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }
        .exclude-colors label {
            font-size: 13px;
            cursor: pointer;
        }
        
        /* NEW Batch Results */
        #batch-results-panel {
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 2px solid #FF9800;
        }
        #batch-status {
            font-size: 16px;
            margin-bottom: 15px;
        }
        #batch-results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 15px;
            max-height: 80vh;
            overflow-y: auto;
            background: #1a1a1a;
            padding: 10px;
            border-radius: 6px;
        }
        #batch-results-grid figure {
            margin: 0;
            background: #222;
            border: 1px solid #444;
            border-radius: 4px;
            overflow: hidden;
        }
        #batch-results-grid img {
            width: 100%;
            height: auto;
            display: block;
            cursor: zoom-in;
            background: #111;
        }
        #batch-results-grid figcaption {
            text-align: center;
            font-size: 13px;
            padding: 8px 0;
            background: #333;
            color: #eee;
            font-weight: bold;
        }
        .detection-details {
            font-size: 11px;
            padding: 8px;
            text-align: left;
            font-weight: normal;
            color: #ccc;
            max-height: 100px;
            overflow-y: auto;
        }
    </style>
</head>
<body>
    <div id="modal" class="modal">
        <span class="close">&times;</span>
        <img class="modal-content" id="modal-img">
        <span class="modal-prev">&#10094;</span>
        <span class="modal-next">&#10095;</span>
    </div>

    <div class="container">
        <h1>🎯 SAR Color Detector (Batch Mode)</h1>
        
        <div id="loading-overlay">
            <div style="text-align: center;">
                <div class="spinner"></div>
                <p id="loading-text">Processing...</p>
            </div>
        </div>

        <div class="upload-controls">
            <button onclick="loadVideo()">📁 Load Video</button>
            <input type="file" id="video-file" accept="video/*" style="display:none;">
            
            <label for="sample-rate" style="font-size: 12px; margin-left: 10px;">Analyze every:</label>
            <input type="number" id="sample-rate" value="10" min="1" max="100" style="width: 50px;" onchange="saveSettings()">
            <label for="sample-rate" style="font-size: 12px;">frames</label>
            
            <label for="pixel-subsample" style="font-size: 12px; margin-left: 10px;">Pixel Subsampling:</label>
            <input type="number" id="pixel-subsample" value="20" min="1" max="100" style="width: 50px;" onchange="saveSettings()">
            <label for="pixel-subsample" style="font-size: 12px;">(1=Max, 50=Fast)</label>
        </div>
        
        <div id="analysis-panel" style="display: none;">
            <h2 style="margin-top: 0;">📊 Video Color Analysis Results</h2>
            <div style="text-align: center; margin-bottom: 20px;">
                <img id="analysis-viz" src="" alt="Analysis visualization" style="max-width: 100%; border-radius: 8px; display: none;">
            </div>
            <div class="analysis-grid">
                <div class="stat-box">
                    <div class="stat-value" id="vivid-percent">0%</div>
                    <div class="stat-label">Vivid Pixels (Sat≥120)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="sat-mean">0</div>
                    <div class="stat-label">Mean Saturation</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="sat-median">0</div>
                    <div class="stat-label">Median Saturation</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="frames-analyzed">0</div>
                    <div class="stat-label">Frames Analyzed</div>
                </div>
            </div>
            <h3 style="margin-top: 20px;">🎨 Vivid Color Distribution</h3>
            <div class="color-groups" id="color-groups"></div>
            <h3 style="margin-top: 20px;">🎯 Top Vivid Hues</h3>
            <div id="top-hues" style="background: #333; padding: 10px; border-radius: 4px; font-size: 12px;"></div>
            <div style="margin-top: 20px;">
                <button onclick="runFullScan()" class="preset-btn" style="width: 100%; padding: 15px; font-size: 16px; background-color: #FF9800;">
                    🤖 Run ML Training & Full Video Scan (All Methods)
                </button>
            </div>
        </div>
        
        <div id="ml-analysis-panel" style="display: none;">
            <h2 style="margin-top: 0;">🤖 Machine Learning Analysis</h2>
            <div style="text-align: center; margin-bottom: 20px;">
                <img id="ml-viz" src="" alt="ML visualization" style="max-width: 100%; border-radius: 8px; display: none;">
            </div>
            <div class="analysis-grid">
                <div class="stat-box">
                    <div class="stat-value" id="pca-variance">0%</div>
                    <div class="stat-label">PCA Variance Explained</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="dbscan-clusters">0</div>
                    <div class="stat-label">DBSCAN Clusters</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="dbscan-outliers">0</div>
                    <div class="stat-label">Outliers (Anomalies)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="kmeans-clusters">0</div>
                    <div class="stat-label">K-Means Clusters</div>
                </div>
            </div>
            <div id="ml-details" style="background: #333; padding: 15px; border-radius: 8px; margin-top: 15px; font-size: 12px;"></div>
        </div>
        
        <div id="batch-results-panel" style="display: none;">
            <h2 style="margin-top: 0;">🎬 Detection Results</h2>
            <p id="batch-status">Scan complete. Found 0 detections.</p>
            <div id="batch-results-grid">
                </div>
        </div>
        
        
        <div class="controls-container">
            <div class="controls">
                <h3>Detection Presets</h3>
                <div class="preset-buttons">
                    <button class="preset-btn" onclick="applyPreset('analysis')">Use Analysis Settings</button>
                    <button class="preset-btn" onclick="applyPreset('default')">Default (Sat≥100)</button>
                    <button class="preset-btn" onclick="applyPreset('strict')">Strict (Sat≥150)</button>
                    <button class="preset-btn" onclick="applyPreset('red')">Red Only</button>
                    <button class="preset-btn" onclick="applyPreset('cyan')">Cyan Only</button>
                    <button class="preset-btn" onclick="applyPreset('all')">All Colors</button>
                </div>
                <h3>General Filtering</h3>
                <div class="control-group">
                    <label>Morphology:</label>
                    <input type="range" id="morph-size" min="0" max="15" step="1" value="5" oninput="updateValue('morph-size-val', this.value); saveSettings();">
                    <span id="morph-size-val">5</span>
                </div>
                <div class="control-group">
                    <label>Exclude Sky:</label>
                    <input type="checkbox" id="exclude-sky" checked onchange="saveSettings();" style="width: 20px;">
                    <span></span>
                </div>
                <h3>Color Exclusions (for HSV)</h3>
                <div class="exclude-colors">
                    <label><input type="checkbox" id="exclude-red" onchange="saveSettings();"> Red</label>
                    <label><input type="checkbox" id="exclude-orange" onchange="saveSettings();"> Orange</label>
                    <label><input type="checkbox" id="exclude-yellow" onchange="saveSettings();"> Yellow</label>
                    <label><input type="checkbox" id="exclude-green" onchange="saveSettings();"> Green</label>
                    <label><input type="checkbox" id="exclude-cyan" onchange="saveSettings();"> Cyan</label>
                    <label><input type="checkbox" id="exclude-blue" onchange="saveSettings();"> Blue</label>
                    <label><input type="checkbox" id="exclude-purple" onchange="saveSettings();"> Purple</label>
                    <label><input type="checkbox" id="exclude-magenta" onchange="saveSettings();"> Magenta</label>
                </div>
            </div>
            
            <div class="controls">
                <h3>HSV Color Range</h3>
                <div class="control-group dual-slider">
                    <label>Hue Range:</label>
                    <input type="range" id="hsv-hue-min" min="0" max="179" value="0" oninput="updateValue('hsv-hue-min-val', this.value); saveSettings();">
                    <input type="range" id="hsv-hue-max" min="0" max="179" value="179" oninput="updateValue('hsv-hue-max-val', this.value); saveSettings();">
                    <span><span id="hsv-hue-min-val">0</span>-<span id="hsv-hue-max-val">179</span></span>
                </div>
                <div class="control-group dual-slider">
                    <label>Sat Range:</label>
                    <input type="range" id="hsv-sat-min" min="0" max="255" value="120" oninput="updateValue('hsv-sat-min-val', this.value); saveSettings();">
                    <input type="range" id="hsv-sat-max" min="0" max="255" value="255" oninput="updateValue('hsv-sat-max-val', this.value); saveSettings();">
                    <span><span id="hsv-sat-min-val">120</span>-<span id="hsv-sat-max-val">255</span></span>
                </div>
                <div class="control-group dual-slider">
                    <label>Val Range:</label>
                    <input type="range" id="hsv-val-min" min="0" max="255" value="30" oninput="updateValue('hsv-val-min-val', this.value); saveSettings();">
                    <input type="range" id="hsv-val-max" min="0" max="255" value="240" oninput="updateValue('hsv-val-max-val', this.value); saveSettings();">
                    <span><span id="hsv-val-min-val">30</span>-<span id="hsv-val-max-val">240</span></span>
                </div>
                <div class="control-group dual-slider">
                    <label>HSV Size:</label>
                    <input type="range" id="hsv-min-size" min="10" max="1000" value="50" oninput="updateValue('hsv-min-size-val', this.value); saveSettings();">
                    <input type="range" id="hsv-max-size" min="1000" max="20000" value="10000" oninput="updateValue('hsv-max-size-val', this.value); saveSettings();">
                    <span><span id="hsv-min-size-val">50</span>-<span id="hsv-max-size-val">10000</span></span>
                </div>
            </div>

            <div class="controls">
                <h3>Saturation Range (High Sat)</h3>
                <div class="control-group">
                    <label>Sat Min:</label>
                    <input type="range" id="sat-sat-min" min="0" max="255" value="200" oninput="updateValue('sat-sat-min-val', this.value); saveSettings();">
                    <span id="sat-sat-min-val">200</span>
                </div>
                <div class="control-group dual-slider">
                    <label>Sat Size:</label>
                    <input type="range" id="sat-min-size" min="10" max="1000" value="50" oninput="updateValue('sat-min-size-val', this.value); saveSettings();">
                    <input type="range" id="sat-max-size" min="1000" max="20000" value="10000" oninput="updateValue('sat-max-size-val', this.value); saveSettings();">
                    <span><span id="sat-min-size-val">50</span>-<span id="sat-max-size-val">10000</span></span>
                </div>
            </div>
            
            <div class="controls">
                <h3>YCrCb Color Range</h3>
                <div class="control-group dual-slider">
                    <label>Cr Range:</label>
                    <input type="range" id="ycrcb-cr-min" min="0" max="255" value="150" oninput="updateValue('ycrcb-cr-min-val', this.value); saveSettings();">
                    <input type="range" id="ycrcb-cr-max" min="0" max="255" value="255" oninput="updateValue('ycrcb-cr-max-val', this.value); saveSettings();">
                    <span><span id="ycrcb-cr-min-val">150</span>-<span id="ycrcb-cr-max-val">255</span></span>
                </div>
                <div class="control-group dual-slider">
                    <label>Cb Range:</label>
                    <input type="range" id="ycrcb-cb-min" min="0" max="255" value="0" oninput="updateValue('ycrcb-cb-min-val', this.value); saveSettings();">
                    <input type="range" id="ycrcb-cb-max" min="0" max="255" value="120" oninput="updateValue('ycrcb-cb-max-val', this.value); saveSettings();">
                    <span><span id="ycrcb-cb-min-val">0</span>-<span id="ycrcb-cb-max-val">120</span></span>
                </div>
                <div class="control-group dual-slider">
                    <label>YCrCb Size:</label>
                    <input type="range" id="ycrcb-min-size" min="10" max="1000" value="50" oninput="updateValue('ycrcb-min-size-val', this.value); saveSettings();">
                    <input type="range" id="ycrcb-max-size" min="1000" max="20000" value="10000" oninput="updateValue('ycrcb-max-size-val', this.value); saveSettings();">
                    <span><span id="ycrcb-min-size-val">50</span>-<span id="ycrcb-max-size-val">10000</span></span>
                </div>
            </div>
            
            <div class="controls">
                <h3>ML Outlier Filtering</h3>
                <div class="control-group">
                    <label>Terrain Filter (StdDevs):</label>
                    <input type="range" id="ml-terrain-stdev" min="1" max="10" step="0.5" value="5" oninput="updateValue('ml-terrain-stdev-val', this.value); saveSettings();">
                    <span id="ml-terrain-stdev-val">5</span>
                </div>
                <div class="control-group dual-slider">
                    <label>ML Size:</label>
                    <input type="range" id="ml-min-size" min="10" max="1000" value="50" oninput="updateValue('ml-min-size-val', this.value); saveSettings();">
                    <input type="range" id="ml-max-size" min="1000" max="20000" value="5000" oninput="updateValue('ml-max-size-val', this.value); saveSettings();">
                    <span><span id="ml-min-size-val">50</span>-<span id="ml-max-size-val">5000</span></span>
                </div>
                <p style="font-size: 12px; color: #aaa;">Note: These settings are used for the "Run Full Video Scan" button.</p>
            </div>
        </div>
    </div>
    
    <script>
        let totalFrames = 0;
        let analysisData = null;
        
        // NEW Modal Globals
        let currentModalImageIndex = 0;
        let modalImageSources = [];

        // --- Save/Load Settings ---
        const settingsKey = 'sarDetectorSettings';
        const controlIds = [
            'morph-size', 'exclude-sky',
            'hsv-hue-min', 'hsv-hue-max', 'hsv-sat-min', 'hsv-sat-max', 'hsv-val-min', 'hsv-val-max', 'hsv-min-size', 'hsv-max-size',
            'sat-sat-min', 'sat-min-size', 'sat-max-size',
            'ycrcb-cr-min', 'ycrcb-cr-max', 'ycrcb-cb-min', 'ycrcb-cb-max', 'ycrcb-min-size', 'ycrcb-max-size',
            'ml-terrain-stdev', 'ml-min-size', 'ml-max-size', // Added ml-terrain-stdev
            'sample-rate', 'pixel-subsample',
            'exclude-red', 'exclude-orange', 'exclude-yellow', 'exclude-green', 'exclude-cyan', 'exclude-blue', 'exclude-purple', 'exclude-magenta'
        ];

        function saveSettings() {
            const settings = {};
            controlIds.forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    if (el.type === 'checkbox') {
                        settings[id] = el.checked;
                    } else {
                        settings[id] = el.value;
                    }
                }
            });
            localStorage.setItem(settingsKey, JSON.stringify(settings));
        }

        function loadSettings() {
            const settings = JSON.parse(localStorage.getItem(settingsKey));
            if (settings) {
                controlIds.forEach(id => {
                    const el = document.getElementById(id);
                    if (el && settings[id] !== undefined) {
                        if (el.type === 'checkbox') {
                            el.checked = settings[id];
                        } else {
                            el.value = settings[id];
                            // Update text span if it exists
                            const spanId = id + '-val';
                            const span = document.getElementById(spanId);
                            if (span) {
                                span.textContent = el.value;
                            }
                        }
                    }
                });
                // Special case for dual sliders
                updateValue('hsv-hue-min-val', settings['hsv-hue-min'] || 0);
                updateValue('hsv-hue-max-val', settings['hsv-hue-max'] || 179);
                updateValue('hsv-sat-min-val', settings['hsv-sat-min'] || 120);
                updateValue('hsv-sat-max-val', settings['hsv-sat-max'] || 255);
                updateValue('hsv-val-min-val', settings['hsv-val-min'] || 30);
                updateValue('hsv-val-max-val', settings['hsv-val-max'] || 240);
                updateValue('hsv-min-size-val', settings['hsv-min-size'] || 50);
                updateValue('hsv-max-size-val', settings['hsv-max-size'] || 10000);
                updateValue('sat-min-size-val', settings['sat-min-size'] || 50);
                updateValue('sat-max-size-val', settings['sat-max-size'] || 10000);
                updateValue('ycrcb-cr-min-val', settings['ycrcb-cr-min'] || 150);
                updateValue('ycrcb-cr-max-val', settings['ycrcb-cr-max'] || 255);
                updateValue('ycrcb-cb-min-val', settings['ycrcb-cb-min'] || 0);
                updateValue('ycrcb-cb-max-val', settings['ycrcb-cb-max'] || 120);
                updateValue('ycrcb-min-size-val', settings['ycrcb-min-size'] || 50);
                updateValue('ycrcb-max-size-val', settings['ycrcb-max-size'] || 10000);
                updateValue('ml-terrain-stdev-val', settings['ml-terrain-stdev'] || 5); // Added
                updateValue('ml-min-size-val', settings['ml-min-size'] || 50);
                updateValue('ml-max-size-val', settings['ml-max-size'] || 5000);
            }
        }
        // --- END Save/Load ---

        function updateValue(id, value) {
            const el = document.getElementById(id);
            if (el) el.textContent = value;
        }
        
        function showLoading(text = "Processing...") {
            document.getElementById('loading-text').textContent = text;
            document.getElementById('loading-overlay').style.display = 'flex';
        }
        
        function hideLoading() {
            document.getElementById('loading-overlay').style.display = 'none';
        }
        
        function loadVideo() {
            document.getElementById('video-file').click();
        }
        
        document.getElementById('video-file').addEventListener('change', function(e) {
            // Guard clause for cancelled file dialog
            if (e.target.files.length === 0) {
                console.log("No file selected.");
                return;
            }
            const formData = new FormData();
            formData.append('video', e.target.files[0]);
            
            // Get sample rate from UI
            const sampleRate = document.getElementById('sample-rate').value;
            formData.append('sample_rate', sampleRate);
            // Get pixel subsample rate
            const subsample = document.getElementById('pixel-subsample').value;
            formData.append('subsample', subsample);

            showLoading('Analyzing video colors... This may take 30-60 seconds...');
            
            fetch('/load_video', {
                method: 'POST',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                hideLoading();
                if (data.success) {
                    totalFrames = data.total_frames;
                    
                    console.log('Analysis data received:', data.analysis);
                    
                    // Display analysis results
                    analysisData = data.analysis;
                    displayAnalysisResults(analysisData);
                    
                    if (!localStorage.getItem(settingsKey)) {
                        document.getElementById('hsv-sat-min').value = 120;
                        updateValue('hsv-sat-min-val', 120);
                    }
                    
                    // REMOVED ALERT
                    
                } else {
                    alert('Error loading video: ' + data.error);
                }
            })
            .catch(err => {
                hideLoading();
                alert('Error: ' + err);
                console.error(err);
            });
        });
        
        function displayAnalysisResults(analysis) {
            try {
                const panel = document.getElementById('analysis-panel');
                panel.style.display = 'block';
                
                const vizImg = document.getElementById('analysis-viz');
                vizImg.src = '/analysis_image?' + new Date().getTime();
                vizImg.style.display = 'block';
                
                document.getElementById('vivid-percent').textContent = analysis.vivid_percent.toFixed(1) + '%';
                document.getElementById('sat-mean').textContent = analysis.sat_mean.toFixed(0);
                document.getElementById('sat-median').textContent = analysis.sat_median.toFixed(0);
                document.getElementById('frames-analyzed').textContent = analysis.frames_analyzed;
                
                const colorGroupsDiv = document.getElementById('color-groups');
                let colorGroupsHtml = '';
                for (const [group, count] of Object.entries(analysis.color_groups)) {
                    if (count > 0) {
                        const percent = (analysis.vivid_pixels > 0 ? (count / analysis.vivid_pixels * 100) : 0).toFixed(1);
                        colorGroupsHtml += `
                            <div class="color-group-item">
                                <strong>${group}</strong><br>
                                ${count.toLocaleString()} pixels (${percent}%)
                            </div>
                        `;
                    }
                }
                colorGroupsDiv.innerHTML = colorGroupsHtml;
                
                const topHuesDiv = document.getElementById('top-hues');
                let topHuesHtml = 'Recommended detection hues:<br><br>';
                for (const [hue, count] of analysis.top_vivid_hues) {
                    const percent = (analysis.vivid_pixels > 0 ? (count / analysis.vivid_pixels * 100) : 0).toFixed(1);
                    const colorName = getColorName(hue);
                    topHuesHtml += `<div style="display: inline-block; margin: 5px; padding: 5px 10px; background: #444; border-radius: 4px;">
                        Hue ${hue} (${colorName}): ${percent}%
                    </div>`;
                }
                topHuesDiv.innerHTML = topHuesHtml;
                
            } catch (error) {
                console.error('Error displaying analysis results:', error);
                alert('Error displaying analysis: ' + error.message);
            }
        }
        
        function getColorName(hue) {
            if (hue <= 15 || hue >= 170) return "Red";
            if (hue <= 30) return "Orange";
            if (hue <= 45) return "Yellow";
            if (hue <= 85) return "Green";
            if (hue <= 115) return "Cyan";
            if (hue <= 145) return "Blue";
            if (hue <= 165) return "Purple";
            return "Magenta";
        }

        function applyPreset(preset) {
            const presets = {
                'analysis': {hsv_hue_min: 0, hsv_hue_max: 179, hsv_sat_min: 120, hsv_sat_max: 255},
                'default': {hsv_hue_min: 0, hsv_hue_max: 179, hsv_sat_min: 100, hsv_sat_max: 255},
                'strict': {hsv_hue_min: 0, hsv_hue_max: 179, hsv_sat_min: 150, hsv_sat_max: 255},
                'red': {hsv_hue_min: 0, hsv_hue_max: 15, hsv_sat_min: 120, hsv_sat_max: 255},
                'cyan': {hsv_hue_min: 86, hsv_hue_max: 115, hsv_sat_min: 120, hsv_sat_max: 255},
                'all': {hsv_hue_min: 0, hsv_hue_max: 179, hsv_sat_min: 0, hsv_sat_max: 255}
            };
            
            const p = presets[preset];
            document.getElementById('hsv-hue-min').value = p.hsv_hue_min;
            document.getElementById('hsv-hue-max').value = p.hsv_hue_max;
            document.getElementById('hsv-sat-min').value = p.hsv_sat_min;
            document.getElementById('hsv-sat-max').value = p.hsv_sat_max;
            
            updateValue('hsv-hue-min-val', p.hsv_hue_min);
            updateValue('hsv-hue-max-val', p.hsv_hue_max);
            updateValue('hsv-sat-min-val', p.hsv_sat_min);
            updateValue('hsv-sat-max-val', p.hsv_sat_max);
            
            saveSettings();
        }
        
        // --- NEW MODAL NAVIGATION LOGIC ---
        function showModalImage(index) {
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modal-img');
            const modalPrev = document.querySelector('.modal-prev');
            const modalNext = document.querySelector('.modal-next');

            if (index < 0 || index >= modalImageSources.length) {
                return; // Out of bounds
            }
            
            currentModalImageIndex = index;
            modalImg.src = modalImageSources[index].src;
            
            // Show/hide arrows
            modalPrev.style.display = (index === 0) ? 'none' : 'block';
            modalNext.style.display = (index === modalImageSources.length - 1) ? 'none' : 'block';
        }

        document.addEventListener('DOMContentLoaded', () => {
            loadSettings();
            
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modal-img');
            const modalPrev = document.querySelector('.modal-prev');
            const modalNext = document.querySelector('.modal-next');

            // Click listener for batch grid
            const container = document.querySelector('.container');
            container.addEventListener('click', function(e) {
                if (e.target.tagName === 'IMG' && e.target.closest('#batch-results-grid')) {
                    if (e.target.src) {
                        // Populate image source list
                        modalImageSources = Array.from(document.querySelectorAll('#batch-results-grid img'));
                        const clickedIndex = modalImageSources.findIndex(img => img.src === e.target.src);
                        
                        if (clickedIndex !== -1) {
                            modal.style.display = 'block';
                            showModalImage(clickedIndex);
                        }
                    }
                }
            });

            // Close button
            const closeBtn = document.querySelector('.modal .close');
            closeBtn.onclick = function() {
                modal.style.display = 'none';
            }
            
            // Click outside
            modal.onclick = function(e) {
                if (e.target === modal) {
                    modal.style.display = 'none';
                }
            }
            
            // Arrow click handlers
            modalPrev.onclick = () => showModalImage(currentModalImageIndex - 1);
            modalNext.onclick = () => showModalImage(currentModalImageIndex + 1);
            
            // Keyboard navigation
            document.addEventListener('keydown', (e) => {
                if (modal.style.display === 'block') {
                    if (e.key === 'ArrowLeft') {
                        showModalImage(currentModalImageIndex - 1);
                    } else if (e.key === 'ArrowRight') {
                        showModalImage(currentModalImageIndex + 1);
                    } else if (e.key === 'Escape') {
                        modal.style.display = 'none';
                    }
                }
            });
        });
        // --- END MODAL LOGIC ---

        function runFullScan() {
            if (!analysisData) {
                alert('Please load and analyze a video first');
                return;
            }
            
            showLoading("Step 1/3: Training ML Models (PCA, DBSCAN)...");
            
            fetch('/ml_analysis', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        throw new Error("Server error: " + text.substring(0, 200));
                    });
                }
                return response.json();
            })
            .then(data => {
                if (!data.success) {
                    throw new Error(data.error); // ML training failed
                }
                
                displayMLResults(data.results);
                
                showLoading("Step 2/3: Scanning video for all detections...");
                
                // --- NEW: Collect ALL params ---
                let excluded_colors = [];
                document.querySelectorAll('.exclude-colors input[type="checkbox"]:checked').forEach(cb => {
                    excluded_colors.push(cb.id.replace('exclude-', ''));
                });

                const scanParams = {
                    // General
                    sample_rate: parseInt(document.getElementById('sample-rate').value),
                    morph_size: parseInt(document.getElementById('morph-size').value),
                    exclude_sky: document.getElementById('exclude-sky').checked,
                    excluded_colors: excluded_colors.map(c => c.charAt(0).toUpperCase() + c.slice(1)),
                    
                    // HSV
                    hsv_hue_min: parseInt(document.getElementById('hsv-hue-min').value),
                    hsv_hue_max: parseInt(document.getElementById('hsv-hue-max').value),
                    hsv_sat_min: parseInt(document.getElementById('hsv-sat-min').value),
                    hsv_sat_max: parseInt(document.getElementById('hsv-sat-max').value),
                    hsv_val_min: parseInt(document.getElementById('hsv-val-min').value),
                    hsv_val_max: parseInt(document.getElementById('hsv-val-max').value),
                    hsv_min_size: parseInt(document.getElementById('hsv-min-size').value),
                    hsv_max_size: parseInt(document.getElementById('hsv-max-size').value),
                    
                    // Saturation
                    sat_sat_min: parseInt(document.getElementById('sat-sat-min').value),
                    sat_min_size: parseInt(document.getElementById('sat-min-size').value),
                    sat_max_size: parseInt(document.getElementById('sat-max-size').value),

                    // YCrCb
                    ycrcb_cr_min: parseInt(document.getElementById('ycrcb-cr-min').value),
                    ycrcb_cr_max: parseInt(document.getElementById('ycrcb-cr-max').value),
                    ycrcb_cb_min: parseInt(document.getElementById('ycrcb-cb-min').value),
                    ycrcb_cb_max: parseInt(document.getElementById('ycrcb-cb-max').value),
                    ycrcb_min_size: parseInt(document.getElementById('ycrcb-min-size').value),
                    ycrcb_max_size: parseInt(document.getElementById('ycrcb-max-size').value),
                    
                    // ML
                    ml_terrain_stdev: parseFloat(document.getElementById('ml-terrain-stdev').value),
                    ml_min_size: parseInt(document.getElementById('ml-min-size').value),
                    ml_max_size: parseInt(document.getElementById('ml-max-size').value)
                };
                // --- END ALL PARAMS ---
                
                return fetch('/run_full_scan', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(scanParams)
                });
            })
            .then(response => {
                if (!response.ok) {
                    return response.text().then(text => {
                        throw new Error("Server error: " + text.substring(0, 200));
                    });
                }
                return response.json();
            })
            .then(data => {
                if (!data.success) {
                    throw new Error(data.error); // Full scan failed
                }
                
                showLoading("Step 3/3: Rendering results...");
                displayBatchResults(data.results);
                hideLoading();
            })
            .catch(err => {
                hideLoading();
                console.error('Error in full scan process:', err);
                alert('Error: ' + err.message);
            });
        }
        
        function displayMLResults(results) {
            const panel = document.getElementById('ml-analysis-panel');
            panel.style.display = 'block';
            
            const vizImg = document.getElementById('ml-viz');
            vizImg.src = '/ml_analysis_image?' + new Date().getTime();
            vizImg.style.display = 'block';
            
            document.getElementById('pca-variance').textContent = (results.pca_variance * 100).toFixed(1) + '%';
            document.getElementById('dbscan-clusters').textContent = results.dbscan_clusters;
            document.getElementById('dbscan-outliers').textContent = results.dbscan_outliers.toLocaleString() + 
                ` (${results.dbscan_outlier_percent.toFixed(1)}%)`;
            document.getElementById('kmeans-clusters').textContent = results.kmeans_clusters;
            
            const detailsDiv = document.getElementById('ml-details');
            let html = '<strong>Analysis Insights:</strong><br><br>';
            html += `<strong>PCA (Principal Component Analysis):</strong><br>`;
            html += `• First ${results.pca_components} components explain ${(results.pca_variance*100).toFixed(1)}% of color variance<br>`;
            html += `• Reduces ${results.original_dimensions}D color space to ${results.pca_components}D<br><br>`;
            
            html += `<strong>DBSCAN (Density-Based Clustering):</strong><br>`;
            html += `• Found ${results.dbscan_clusters} dense color clusters (terrain types)<br>`;
            html += `• Identified ${results.dbscan_outliers.toLocaleString()} outlier pixels (${results.dbscan_outlier_percent.toFixed(1)}%)<br>`;
            html += `• Outliers likely represent vehicles/equipment/people<br><br>`;
            
            html += `<strong>K-Means Clustering:</strong><br>`;
            html += `• Segmented colors into ${results.kmeans_clusters} groups<br>`;
            
            detailsDiv.innerHTML = html;
            
            panel.scrollIntoView({ behavior: 'smooth' });
        }
        
        function displayBatchResults(results) {
            const panel = document.getElementById('batch-results-panel');
            const grid = document.getElementById('batch-results-grid');
            const status = document.getElementById('batch-status');
            
            grid.innerHTML = ''; // Clear previous results
            
            if (results.length === 0) {
                status.textContent = 'Scan complete. No detections found matching your filter criteria.';
                panel.style.display = 'block';
                return;
            }
            
            let totalDetections = 0;
            let html = '';
            
            for (const item of results) {
                totalDetections += item.detections.length;
                
                let detailsHtml = '';
                item.detections.forEach(d => {
                    detailsHtml += `• ${d.label} - ${d.area}px at (${d.x}, ${d.y})<br>`;
                });
            
                html += `
                    <figure>
                        <img src="data:image/jpeg;base64,${item.image_b64}" alt="Detection on frame ${item.frame_num}">
                        <figcaption>
                            Frame: ${item.frame_num} | Detections: ${item.detections.length}
                            <div class="detection-details">
                                ${detailsHtml}
                            </div>
                        </figcaption>
                    </figure>
                `;
            }
            
            status.textContent = `Scan complete. Found ${totalDetections} detections across ${results.length} frames.`;
            grid.innerHTML = html;
            panel.style.display = 'block';
            panel.scrollIntoView({ behavior: 'smooth' });
        }
        
    </script>
</body>
</html>
    """
    return html

@app.route('/analysis_image')
def analysis_image():
    """Generate and serve the analysis visualization"""
    global video_path, analysis_results
    
    if video_path is None or analysis_results is None:
        return "No analysis available", 404
    
    # Generate visualization similar to the unbiased one
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#2a2a2a')
    
    # Get sample data for visualization
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    all_hues = []
    all_sats = []
    all_vals = []
    colors_rgb = []
    
    # Sample frames
    for frame_num in np.linspace(0, total_frames-1, 20, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w = hsv.shape[:2]
        for y in range(0, h, 15):
            for x in range(0, w, 15):
                all_hues.append(hsv[y, x, 0])
                all_sats.append(hsv[y, x, 1])
                all_vals.append(hsv[y, x, 2])
                colors_rgb.append([rgb[y, x, 0]/255, rgb[y, x, 1]/255, rgb[y, x, 2]/255])
    
    cap.release()
    
    all_hues = np.array(all_hues)
    all_sats = np.array(all_sats)
    all_vals = np.array(all_vals)
    colors_rgb = np.array(colors_rgb)
    
    # Plot 1: Hue Distribution
    axes[0, 0].hist(all_hues, bins=180, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Hue (0-179)', color='white')
    axes[0, 0].set_ylabel('Frequency', color='white')
    axes[0, 0].set_title('Hue Distribution', color='white', fontweight='bold')
    axes[0, 0].tick_params(colors='white')
    axes[0, 0].set_facecolor('#1a1a1a')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Hue vs Saturation
    axes[0, 1].scatter(all_hues, all_sats, c=colors_rgb, s=1, alpha=0.3)
    axes[0, 1].set_xlabel('Hue', color='white')
    axes[0, 1].set_ylabel('Saturation', color='white')
    axes[0, 1].set_title('Hue vs Saturation (actual colors)', color='white', fontweight='bold')
    axes[0, 1].tick_params(colors='white')
    axes[0, 1].set_facecolor('#1a1a1a')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Saturation vs Value
    axes[0, 2].scatter(all_sats, all_vals, c=colors_rgb, s=1, alpha=0.3)
    axes[0, 2].set_xlabel('Saturation', color='white')
    axes[0, 2].set_ylabel('Value', color='white')
    axes[0, 2].set_title('Saturation vs Value (actual colors)', color='white', fontweight='bold')
    axes[0, 2].tick_params(colors='white')
    axes[0, 2].set_facecolor('#1a1a1a')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Saturation Distribution
    axes[1, 0].hist(all_sats, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(120, color='lime', linestyle='--', linewidth=2, label='Threshold (120)')
    axes[1, 0].set_xlabel('Saturation', color='white')
    axes[1, 0].set_ylabel('Frequency', color='white')
    axes[1, 0].set_title('Saturation Distribution', color='white', fontweight='bold')
    axes[1, 0].tick_params(colors='white')
    axes[1, 0].set_facecolor('#1a1a1a')
    axes[1, 0].legend(facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Color Groups Bar Chart
    color_groups = analysis_results['color_groups']
    group_names = list(color_groups.keys())
    group_values = list(color_groups.values())
    
    bars = axes[1, 1].bar(range(len(group_names)), group_values, color='skyblue', edgecolor='black')
    axes[1, 1].set_xticks(range(len(group_names)))
    axes[1, 1].set_xticklabels([g.split('(')[0].strip() for g in group_names], rotation=45, ha='right', color='white')
    axes[1, 1].set_ylabel('Pixel Count', color='white')
    axes[1, 1].set_title('Vivid Color Groups (Sat≥120)', color='white', fontweight='bold')
    axes[1, 1].tick_params(colors='white')
    axes[1, 1].set_facecolor('#1a1a1a')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Statistics Text
    axes[1, 2].axis('off')
    stats_text = f"""
ANALYSIS RESULTS

Total Pixels: {analysis_results['total_pixels']:,}
Vivid Pixels: {analysis_results['vivid_pixels']:,}
Vivid %: {analysis_results['vivid_percent']:.1f}%

SATURATION:
  Mean: {analysis_results['sat_mean']:.1f}
  Median: {analysis_results['sat_median']:.1f}
  25th: {analysis_results['sat_p25']:.1f}
  75th: {analysis_results['sat_p75']:.1f}

TOP VIVID HUES:
"""
    
    top_hues = analysis_results['top_vivid_hues'][:5]
    if analysis_results['vivid_pixels'] > 0:
        for hue, count in top_hues:
            pct = count / analysis_results['vivid_pixels'] * 100
            stats_text += f"   Hue {hue}: {pct:.1f}%\n"
    else:
        stats_text += "   (No vivid pixels found)\n"

    
    axes[1, 2].text(0.1, 0.95, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    color='white',
                    bbox=dict(boxstyle='round', facecolor='#333', alpha=0.8))
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#2a2a2a')
    buf.seek(0)
    plt.close()
    
    return send_file(buf, mimetype='image/png')

@app.route('/load_video', methods=['POST'])
def load_video():
    """Load a video file and run analysis"""
    global video_path, total_frames, fps, frame_cache, analysis_results, ml_results_cache
    
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file'})
        
        video_file = request.files['video']
        
        try:
            sample_rate = int(request.form.get('sample_rate', 10))
            if sample_rate < 1:
                sample_rate = 1
        except ValueError:
            sample_rate = 10

        try:
            subsample = int(request.form.get('subsample', 20))
            if subsample < 1:
                subsample = 1
        except ValueError:
            subsample = 20

        video_path = 'uploaded_video_temp.mp4'
        video_file.save(video_path)
        
        frame_cache = {}
        ml_results_cache = None 
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Could not open video'})
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        print(f"✅ Video loaded: {total_frames} frames @ {fps:.1f} fps") 
        
        analysis_results = analyze_video_colors(video_path, sample_rate=sample_rate, subsample=subsample, min_saturation=120)
        
        return jsonify({
            'success': True,
            'total_frames': total_frames,
            'fps': fps,
            'analysis': analysis_results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/run_full_scan', methods=['POST'])
def run_full_scan():
    """
    Runs a full video scan for ALL detection methods
    based on user-provided filter settings.
    """
    global video_path, total_frames, ml_results_cache
    
    try:
        if video_path is None or ml_results_cache is None:
            return jsonify({'success': False, 'error': 'Video not loaded or ML not trained'})
            
        params = request.json
        sample_rate = int(params.get('sample_rate', 10))

        print(f"\n🎬 Starting full video scan (All Methods)...")
        print(f"   Settings: SampleRate={sample_rate}, Morph={params['morph_size']}, ML-StdDevs={params['ml_terrain_stdev']}")
        
        results = []
        frame_indices = range(0, total_frames, sample_rate)
        
        for i, frame_num in enumerate(frame_indices):
            frame = get_frame(frame_num)
            if frame is None:
                continue

            # Process the frame for all detection types
            frame_out, detections = process_frame_for_batch(frame, params)
            
            # If any detections were found, save the result
            if len(detections) > 0:
                print(f"   -> Detection found on frame {frame_num} ({len(detections)} contours)")
                
                # Encode image
                image_b64 = encode_image(frame_out, quality=80)
                
                results.append({
                    'frame_num': frame_num,
                    'image_b64': image_b64,
                    'detections': detections
                })
            
            if i % 20 == 0:
                print(f"   ...scanned {i}/{len(frame_indices)} frames...")
                
        print(f"✅ Scan complete. Found {len(results)} frames with detections.")
        
        return jsonify({'success': True, 'results': results})
        
    except Exception as e:
        print(f"Error in /run_full_scan: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/ml_analysis', methods=['POST'])
def ml_analysis():
    """Run ML analysis on video colors"""
    global video_path, analysis_results, ml_results_cache
    
    try:
        if video_path is None:
            return jsonify({'success': False, 'error': 'No video loaded'})
        
        print("\n🤖 Running IMPROVED ML analysis (terrain baseline)...")
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        color_samples = []
        frame_indices = np.linspace(0, total_frames-1, min(100, total_frames), dtype=int)
        
        print(f"   Sampling {len(frame_indices)} frames across video...")
        
        for frame_num in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            
            h, w = hsv.shape[:2]
            for y in range(0, h, 30):
                for x in range(0, w, 30):
                    hue = hsv[y, x, 0]
                    sat = hsv[y, x, 1]
                    val = hsv[y, x, 2]
                    l = lab[y, x, 0]
                    a = lab[y, x, 1]
                    b = lab[y, x, 2]
                    
                    if sat > 15 and 20 < val < 240:
                        color_samples.append([hue, sat, val, l, a, b])
        
        cap.release()
        
        if len(color_samples) < 50:
            return jsonify({'success': False, 'error': 'Not enough color samples (need at least 50).'})
            
        color_samples = np.array(color_samples)
        
        max_ml_samples = 100000
        if len(color_samples) > max_ml_samples:
            print(f"   Subsampling {len(color_samples)} down to {max_ml_samples} for ML training...")
            indices = np.random.choice(len(color_samples), max_ml_samples, replace=False)
            color_samples = color_samples[indices]
            
        print(f"   Using {len(color_samples):,} samples for ML training.")
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(color_samples)
        
        print("   Running PCA...")
        pca_components = min(3, features_scaled.shape[1], len(features_scaled))
        if pca_components < 1:
            return jsonify({'success': False, 'error': 'Not enough data for PCA.'})
            
        pca = PCA(n_components=pca_components)
        pca_result = pca.fit_transform(features_scaled)
        pca_variance = pca.explained_variance_ratio_.sum()
        
        print("   Running DBSCAN (EXTREMELY STRICT)...")
        dbscan = DBSCAN(eps=0.2, min_samples=200)
        dbscan_labels = dbscan.fit_predict(pca_result)
        
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_outliers = list(dbscan_labels).count(-1)
        outlier_percent = (n_outliers / len(dbscan_labels)) * 100
        
        print("   Running K-Means...")
        kmeans_clusters = min(5, len(pca_result) - 1)
        if kmeans_clusters < 1:
            return jsonify({'success': False, 'error': 'Not enough data for K-Means.'})
            
        kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(pca_result)
        
        outlier_kmeans_cluster_label = None
        if kmeans_clusters > 0:
            cluster_counts = Counter(kmeans_labels)
            smallest_cluster = cluster_counts.most_common()[-1]
            outlier_kmeans_cluster_label = smallest_cluster[0]

        
        core_sample_indices = dbscan.core_sample_indices_ if hasattr(dbscan, 'core_sample_indices_') else None
        core_samples = pca_result[core_sample_indices] if core_sample_indices is not None else np.array([])
        
        # --- NEW: Store terrain_std, not radius ---
        if len(core_samples) > 0:
            terrain_centroid = np.mean(core_samples, axis=0)
            terrain_std = np.std(core_samples, axis=0)
            print(f"   ✅ Terrain baseline: {len(core_samples)} samples, mean_std={np.mean(terrain_std):.3f}")
        else:
            terrain_centroid = np.mean(pca_result, axis=0)
            terrain_std = np.std(pca_result, axis=0)
            print(f"   ⚠️  No core samples, using full dataset")
        # --- END NEW ---
        
        outlier_mask = dbscan_labels == -1
        outlier_samples = color_samples[outlier_mask]
        outlier_sat_mean = np.mean(outlier_samples[:, 1]) if len(outlier_samples) > 0 else 180
        outlier_sat_threshold = max(outlier_sat_mean * 0.95, 150)
        print(f"   ✅ Outlier sat threshold: {outlier_sat_threshold:.1f}")
        
        max_core_samples_for_detection = 5000
        if len(core_samples) > max_core_samples_for_detection:
            core_samples_subset = core_samples[np.random.choice(len(core_samples), max_core_samples_for_detection, replace=False)]
        else:
            core_samples_subset = core_samples
        
        ml_results_cache = {
            'color_samples': color_samples,
            'pca_result': pca_result,
            'dbscan_labels': dbscan_labels,
            'kmeans_labels': kmeans_labels,
            'pca': pca,
            'scaler': scaler,
            'dbscan': dbscan,
            'kmeans': kmeans,
            'core_samples': core_samples,
            'core_samples_subset': core_samples_subset,
            'outlier_kmeans_cluster_label': outlier_kmeans_cluster_label,
            'terrain_centroid': terrain_centroid,
            'terrain_std': terrain_std, # STORED STD, NOT RADIUS
            'outlier_sat_threshold': outlier_sat_threshold
        }
        
        print(f"   PCA: {pca_variance*100:.1f}% variance")
        print(f"   DBSCAN: {n_clusters} clusters, {n_outliers:,} outliers ({outlier_percent:.1f}%)")
        print(f"   K-Means: {kmeans_clusters} clusters")
        
        results = {
            'pca_variance': float(pca_variance),
            'pca_components': pca_components,
            'original_dimensions': features_scaled.shape[1],
            'dbscan_clusters': int(n_clusters),
            'dbscan_outliers': int(n_outliers),
            'dbscan_outlier_percent': float(outlier_percent),
            'kmeans_clusters': kmeans_clusters
        }
        
        return jsonify({'success': True, 'results': results})
    
    except Exception as e:
        print(f"Error in /ml_analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/ml_analysis_image')
def ml_analysis_image():
    """Generate ML analysis visualization"""
    global ml_results_cache
    
    if 'ml_results_cache' not in globals() or ml_results_cache is None:
        return "No ML analysis available", 404
    
    data = ml_results_cache
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#2a2a2a')
    
    pca_result = data['pca_result']
    dbscan_labels = data['dbscan_labels']
    kmeans_labels = data['kmeans_labels']
    color_samples = data['color_samples']
    
    # Create RGB colors for visualization
    colors_rgb = np.zeros((len(color_samples), 3))
    for i, sample in enumerate(color_samples):
        h, s, v = sample[0], sample[1], sample[2]
        hsv_pixel = np.uint8([[[h, s, v]]])
        rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0][0]
        colors_rgb[i] = rgb_pixel / 255.0
    
    # Plot 1: PCA - PC1 vs PC2 colored by actual color
    axes[0, 0].scatter(pca_result[:, 0], pca_result[:, 1], c=colors_rgb, s=1, alpha=0.5)
    axes[0, 0].set_xlabel('PC1', color='white')
    axes[0, 0].set_ylabel('PC2', color='white')
    axes[0, 0].set_title('PCA: PC1 vs PC2 (actual colors)', color='white', fontweight='bold')
    axes[0, 0].tick_params(colors='white')
    axes[0, 0].set_facecolor('#1a1a1a')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: PCA - PC1 vs PC3
    if pca_result.shape[1] > 2:
        axes[0, 1].scatter(pca_result[:, 0], pca_result[:, 2], c=colors_rgb, s=1, alpha=0.5)
        axes[0, 1].set_xlabel('PC1', color='white')
        axes[0, 1].set_ylabel('PC3', color='white')
        axes[0, 1].set_title('PCA: PC1 vs PC3 (actual colors)', color='white', fontweight='bold')
    else:
        axes[0, 1].text(0.5, 0.5, 'Only 2 PCA Components', horizontalalignment='center', verticalalignment='center', transform=axes[0, 1].transAxes, color='white')
        axes[0, 1].set_title('PCA: PC1 vs PC3', color='white', fontweight='bold')
    axes[0, 1].tick_params(colors='white')
    axes[0, 1].set_facecolor('#1a1a1a')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: PCA Variance
    variance = data['pca'].explained_variance_ratio_
    axes[0, 2].bar(range(len(variance)), variance * 100, color='skyblue', edgecolor='white')
    axes[0, 2].set_xlabel('Component', color='white')
    axes[0, 2].set_ylabel('Variance Explained (%)', color='white')
    axes[0, 2].set_title('PCA Variance Explained', color='white', fontweight='bold')
    axes[0, 2].tick_params(colors='white')
    axes[0, 2].set_facecolor('#1a1a1a')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: DBSCAN clusters
    scatter = axes[1, 0].scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=dbscan_labels, cmap='tab10', s=1, alpha=0.5)
    axes[1, 0].set_xlabel('PC1', color='white')
    axes[1, 0].set_ylabel('PC2', color='white')
    axes[1, 0].set_title('DBSCAN Clustering (outliers in purple)', color='white', fontweight='bold')
    axes[1, 0].tick_params(colors='white')
    axes[1, 0].set_facecolor('#1a1a1a')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: K-Means clusters
    scatter = axes[1, 1].scatter(pca_result[:, 0], pca_result[:, 1], 
                                c=kmeans_labels, cmap='viridis', s=1, alpha=0.5)
    axes[1, 1].set_xlabel('PC1', color='white')
    axes[1, 1].set_ylabel('PC2', color='white')
    axes[1, 1].set_title(f'K-Means Clustering ({data["kmeans"].n_clusters} clusters)', color='white', fontweight='bold')
    axes[1, 1].tick_params(colors='white')
    axes[1, 1].set_facecolor('#1a1a1a')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Outliers only (DBSCAN)
    outlier_mask = dbscan_labels == -1
    if np.sum(outlier_mask) > 0:
        outlier_colors = colors_rgb[outlier_mask]
        outlier_pca = pca_result[outlier_mask]
        axes[1, 2].scatter(outlier_pca[:, 0], outlier_pca[:, 1], 
                           c=outlier_colors, s=3, alpha=0.6)
        axes[1, 2].set_xlabel('PC1', color='white')
        axes[1, 2].set_ylabel('PC2', color='white')
        axes[1, 2].set_title(f'DBSCAN Outliers Only (n={np.sum(outlier_mask):,})', 
                           color='white', fontweight='bold')
    else:
        axes[1, 2].text(0.5, 0.5, 'No Outliers Found', horizontalalignment='center', verticalalignment='center', transform=axes[1, 2].transAxes, color='white')
        axes[1, 2].set_title('DBSCAN Outliers Only', color='white', fontweight='bold')
    axes[1, 2].tick_params(colors='white')
    axes[1, 2].set_facecolor('#1a1a1a')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#2a2a2a')
    buf.seek(0)
    plt.close()
    
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    print("="*80)
    print("🎯 SAR DETECTOR (BATCH MODE)")
    print("="*80)
    print("\nStarting web server...")
    print("\n📡 Open your browser to: http://localhost:5000")
    print("\nFeatures:")
    print("   • Automatically analyzes video on upload")
    print("   • Displays color statistics and recommendations")
    print("   • One-click ML training and full-video batch scan (ALL methods)")
    print("   • Presents all detections in a scrollable grid")
    print("   • Modal lightbox with image navigation (arrows)")
    print("   • Adjustable 'Terrain Filter' for ML noise control")
    print("\n Press Ctrl+C to stop")
    print("="*80)
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
