#!/usr/bin/env python3
# subtitle_extractor_ocr.py - Detect subtitle changes, OCR keyframes, generate SRT

import cv2
import numpy as np
import os
import tempfile
import gradio as gr
from tqdm import tqdm
import datetime
import configparser
import logging
import base64
import io
import json
import shutil
import math
import subprocess
from PIL import Image
from openai import OpenAI
import anthropic
import pysubs2

# --- Constants & Global Setup ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE_NAME = "config.ini"
PROMPT_FILE_NAME = "prompt.txt"
KEYFRAMES_SUBDIR = "keyframes"
COMPOSITE_KEYFRAMES_SUBDIR = "composite_keyframes"
COMPOSITE_BATCH_SIZE = 6 # Number of keyframes per composite image (1x6 vertical grid)
COMPOSITE_BORDER_SIZE = 2 # Pixels for white border between stacked images

# --- AI Model Constants ---
MODEL_CLAUDE_45_SONNET = "claude-sonnet-4-5-20250929"
MODEL_CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
MODEL_CLAUDE_37_SONNET = "claude-3-7-sonnet-20250219"
MODEL_GPT5_CHAT = "gpt-5-chat-latest"
MODEL_GPT41 = "gpt-4.1-2025-04-14"
MODEL_GPT41_MINI = "gpt-4.1-mini"
MODEL_GPT4O = "gpt-4o"
MODEL_O4_MINI = "o4-mini"

# --- Configuration Handling ---
def get_config_path():
    return os.path.join(APP_DIR, CONFIG_FILE_NAME)

def get_api_keys():
    config_path = get_config_path()
    config = configparser.ConfigParser()
    keys = {
        'openai': None,
        'anthropic': None
    }
    
    if os.path.exists(config_path):
        config.read(config_path)
        keys['openai'] = config.get('Credentials', 'openai_api_key', fallback=None)
        keys['anthropic'] = config.get('Credentials', 'anthropic_api_key', fallback=None)
    
    return keys

def get_api_key():
    """Legacy function for backward compatibility"""
    keys = get_api_keys()
    return keys['openai']

def save_api_keys(openai_api_key=None, anthropic_api_key=None):
    config_path = get_config_path()
    config = configparser.ConfigParser()
    
    if os.path.exists(config_path):
        config.read(config_path)
    
    if 'Credentials' not in config:
        config['Credentials'] = {}
    
    if openai_api_key is not None:
        config['Credentials']['openai_api_key'] = openai_api_key
    
    if anthropic_api_key is not None:
        config['Credentials']['anthropic_api_key'] = anthropic_api_key
    
    with open(config_path, 'w') as configfile:
        config.write(configfile)

def save_api_key(api_key):
    """Legacy function for backward compatibility"""
    save_api_keys(openai_api_key=api_key)

# --- Logging Setup ---
def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    return logger

# --- Output Directory ---
def create_process_output_dir(base_output_dir_name="output"):
    """Creates the output directory and necessary subdirs relative to the app.py script."""
    base_output_dir = os.path.join(APP_DIR, base_output_dir_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    process_dir = os.path.join(base_output_dir, f"process_{timestamp}")
    keyframes_dir = os.path.join(process_dir, KEYFRAMES_SUBDIR)
    composite_keyframes_dir = os.path.join(process_dir, COMPOSITE_KEYFRAMES_SUBDIR)
    os.makedirs(keyframes_dir, exist_ok=True)
    os.makedirs(composite_keyframes_dir, exist_ok=True)
    print(f"DEBUG: Absolute process directory created: {os.path.abspath(process_dir)}")
    print(f"DEBUG: Absolute keyframes directory created: {os.path.abspath(keyframes_dir)}")
    print(f"DEBUG: Absolute composite keyframes directory created: {os.path.abspath(composite_keyframes_dir)}")
    return process_dir, keyframes_dir, composite_keyframes_dir

# --- Timestamp Formatting ---
def format_srt_timestamp(milliseconds):
    seconds, milliseconds = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

# ---------- VideoWriter helper ----------
def writer(path, fps, size):
    size = (int(size[0]), int(size[1]))
    for fourcc in ("mp4v", "XVID", "avc1"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, size, True)
        if w.isOpened(): return w
    raise IOError(f"no working codec (mp4v/XVID/avc1 failed for path {path})")

# ---------- Image Processing & Batching ----------
def resize_frame(frame, target_width=512):
    h, w = frame.shape[:2]
    if w == 0: return frame
    ratio = target_width / w
    target_height = int(h * ratio)
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

def burn_timestamp_on_image(image, timestamp_ms, keyframe_path=None, font_scale=0.6, thickness=1):
    """Burns HH:MM:SS,ms timestamp and keyframe name onto the image without covering subtitle content."""
    timestamp_str = format_srt_timestamp(timestamp_ms)
    
    # Add keyframe name if provided
    if keyframe_path:
        keyframe_name = os.path.basename(keyframe_path).split('.')[0]  # Get filename without extension
        timestamp_str = f"{timestamp_str} - {keyframe_name}"
    
    # Create a new image with extra space at the top for the timestamp and name
    h, w = image.shape[:2]
    extra_space = 30  # Add 30 pixels of extra space at the top
    new_h = h + extra_space
    
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Grayscale image
        new_image = np.zeros((new_h, w), dtype=image.dtype)
        new_image[extra_space:, :] = image
        new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
    else:
        # Color image
        new_image = np.zeros((new_h, w, 3), dtype=image.dtype)
        new_image[extra_space:, :] = image
    
    # Position text in the extra space at the top
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)  # Red
    (text_width, text_height), baseline = cv2.getTextSize(timestamp_str, font, font_scale, thickness)
    
    # Center the text horizontally in the extra space
    org = ((w - text_width) // 2, (extra_space + text_height) // 2)
    
    # Add a black background behind the text for better visibility
    bg_top_left = (org[0]-3, org[1]-text_height-3)
    bg_bottom_right = (org[0]+text_width+3, org[1]+baseline+3)
    bg_top_left = (max(0, bg_top_left[0]), max(0, bg_top_left[1]))

    # Draw the text on the new image
    cv2.rectangle(new_image, bg_top_left, bg_bottom_right, (0,0,0), -1)
    cv2.putText(new_image, timestamp_str, org, font, font_scale, color, thickness, cv2.LINE_AA)
    
    return new_image

def create_composite_image(batch_keyframes_info, composite_save_path, target_width=512, logger=None):
    """Creates a composite image by stacking keyframes vertically with borders."""
    images_with_ts = []
    max_w = 0
    total_h_no_borders = 0
    processed_image_paths = []

    for info in batch_keyframes_info:
        try:
            img = cv2.imread(info['absolute_path'])
            if img is None:
                if logger: logger.warning(f"Failed to read keyframe image: {info['absolute_path']}")
                continue
            resized_img = resize_frame(img, target_width)
            img_with_ts = burn_timestamp_on_image(resized_img, info['timestamp_ms'], info['relative_path'])
            images_with_ts.append(img_with_ts)
            h, w = img_with_ts.shape[:2]
            max_w = max(max_w, w)
            total_h_no_borders += h
            processed_image_paths.append(info['relative_path'])
        except Exception as e:
            if logger: logger.error(f"Error processing keyframe {info.get('absolute_path', 'N/A')} for composite: {e}")

    if not images_with_ts:
        if logger: logger.warning("No valid images found in batch to create composite.")
        return False, []

    num_images = len(images_with_ts)
    total_border_height = max(0, num_images - 1) * COMPOSITE_BORDER_SIZE
    final_composite_h = total_h_no_borders + total_border_height

    # Ensure all images have the same width (max_w)
    final_images = []
    current_total_h_check = 0 # Recalculate height based on potentially padded images
    for img in images_with_ts:
        h, w = img.shape[:2]
        if w != max_w:
            canvas = np.zeros((h, max_w, 3), dtype=np.uint8)
            x_offset = (max_w - w) // 2
            canvas[:, x_offset:x_offset+w] = img
            final_images.append(canvas)
            current_total_h_check += h
        else:
            final_images.append(img)
            current_total_h_check += h

    # Recalculate final height based on actual image heights after padding
    final_composite_h = current_total_h_check + max(0, len(final_images) - 1) * COMPOSITE_BORDER_SIZE
    composite_image = np.zeros((final_composite_h, max_w, 3), dtype=np.uint8) # Use black background

    # Stack images vertically with borders
    current_y = 0
    for i, img in enumerate(final_images):
        h = img.shape[0]
        if current_y + h > composite_image.shape[0]: # Boundary check
             if logger: logger.warning(f"Image {i} exceeds composite height, skipping.")
             continue
        composite_image[current_y:current_y+h, :] = img
        current_y += h
        # Add border below image, except for the last one
        if i < len(final_images) - 1:
            border_y_start = current_y
            border_y_end = current_y + COMPOSITE_BORDER_SIZE
            if border_y_end <= composite_image.shape[0]: # Boundary check for border
                 composite_image[border_y_start:border_y_end, :] = (255, 255, 255) # White border
            current_y += COMPOSITE_BORDER_SIZE

    try:
        cv2.imwrite(composite_save_path, composite_image)
        if logger: logger.info(f"Saved composite keyframe image (1x{len(final_images)} with borders): {composite_save_path}")
        return True, processed_image_paths
    except Exception as e:
        if logger: logger.error(f"Failed to save composite keyframe image {composite_save_path}: {e}")
        return False, []


def frames_are_different_percentage(frame1, frame2, roi_area, threshold_percent=1.0, logger=None):
    """Compares two frames using percentage of changed WHITE pixels in the ROI."""
    if frame1 is None or frame2 is None: return frame1 is not frame2
    if frame1.shape != frame2.shape: return True
    
    # Convert frames to grayscale if they're not already
    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        frame1_gray = frame1
        frame2_gray = frame2
    
    # Create masks for white pixels in both frames (pixels with value > 200)
    white_mask1 = (frame1_gray > 200).astype(np.uint8)
    white_mask2 = (frame2_gray > 200).astype(np.uint8)
    
    # Calculate the difference between white pixel masks
    diff_mask = cv2.absdiff(white_mask1, white_mask2)
    
    # Count the number of different white pixels
    diff_white_count = np.count_nonzero(diff_mask)
    
    # Calculate the percentage of changed white pixels relative to the ROI area
    changed_percent = (diff_white_count / roi_area) * 100 if roi_area > 0 else 0
    
    is_diff = changed_percent > threshold_percent
    if logger and is_diff:
        logger.info(f"White pixel change detected: {changed_percent:.2f}% (threshold: {threshold_percent:.2f}%)")
    
    return is_diff

def extract_subtitle_keyframes(
    src: str,
    process_dir: str,
    keyframes_dir: str,
    composite_keyframes_dir: str,
    logger: logging.Logger,
    white_lvl: int = 240,
    tol: int = 100,
    keep_ratio_h: float = 0.15,
    side_ratio: float = 0.20,
    min_blob: int = 1,
    change_percent_threshold: float = 1.0,
    target_frame_width: int = 512,
    save_processed_video: bool = False
) -> tuple[list, list, str | None, list]:
    """
    Processes video, detects subtitle changes, saves individual keyframes,
    creates composite keyframes (1xN vertical stack with borders), and returns data.
    Returns: (list_of_individual_keyframes_info, list_of_time_segments_with_ref, path_to_processed_video | None, list_of_composite_image_paths)
    """
    # Initialize keyframe counter for sequential numbering
    keyframe_counter = 1
    
    logger.info(f"--- Starting Keyframe Extraction ---")
    logger.info(f"Source Video: {src}")
    logger.info(f"Process Directory (Absolute): {os.path.abspath(process_dir)}")
    logger.info(f"Individual Keyframes Directory (Absolute): {os.path.abspath(keyframes_dir)}")
    logger.info(f"Composite Keyframes Directory (Absolute): {os.path.abspath(composite_keyframes_dir)}")
    logger.info(f"Save Processed Video: {save_processed_video}")
    logger.info(f"Using Percentage Change Detection Threshold: {change_percent_threshold:.2f}%")
    logger.info(f"Parameters: white_lvl={white_lvl}, tol={tol}, keep_ratio_h={keep_ratio_h}, side_ratio={side_ratio}, min_blob={min_blob}, target_width={target_frame_width}")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error("Failed to open video file.")
        raise IOError("bad video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_ms = (total_frames / fps) * 1000 if total_frames > 0 and fps > 0 else 0
    logger.info(f"Video properties: {w}x{h} @ {fps:.2f} fps, {total_frames} frames, duration ~{duration_ms/1000:.2f}s")

    roi_h = int(h * keep_ratio_h)
    y0 = h - roi_h
    x0 = int(w * side_ratio)
    roi_w = w - 2 * x0
    if roi_w <= 0 or roi_h <= 0:
        logger.error(f"Calculated ROI dimensions are invalid ({roi_w}x{roi_h}). Adjust keep_ratio_h or side_ratio.")
        cap.release()
        raise ValueError("Invalid ROI dimensions – crop width or height ≤ 0")
    logger.info(f"Subtitle ROI: x={x0}, y={y0}, width={roi_w}, height={roi_h}")
    roi_area = roi_w * roi_h

    processed_video_writer = None
    processed_video_path = None
    if save_processed_video:
        processed_video_filename = "processed_video.mp4"
        processed_video_path = os.path.join(process_dir, processed_video_filename)
        try:
            processed_video_writer = writer(processed_video_path, fps, (roi_w, roi_h))
            logger.info(f"Initialized processed video writer to save at: {processed_video_path}")
        except Exception as e:
            logger.error(f"Failed to initialize processed video writer: {e}")
            processed_video_writer = None
            processed_video_path = None

    individual_keyframes_info = []
    time_segments = []

    last_out_frame = None
    current_segment_start_ms = 0.0
    frame_count = 0

    pbar = tqdm(total=total_frames, unit="fr", ncols=70, desc="Processing frames")

    while True:
        ok, frame = cap.read()
        if not ok:
            logger.info("End of video stream reached.")
            break

        current_frame_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or (frame_count * 1000.0 / fps)
        frame_count += 1
        pbar.update(1)

        roi = frame[y0 : y0 + roi_h, x0 : x0 + roi_w]
        if roi.size == 0:
            logger.warning(f"Frame {frame_count}: ROI is empty, skipping.")
            continue

        b, g, r = cv2.split(roi.astype(np.int16))
        candidates = ((r >= white_lvl) & (g >= white_lvl) & (b >= white_lvl) &
                      (np.abs(r - g) <= tol) & (np.abs(g - b) <= tol) &
                      (np.abs(r - b) <= tol)).astype(np.uint8)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
        clean_mask = np.zeros_like(candidates)
        has_subtitle_pixels = False
        for i in range(1, num):
            # Changed to use max_blob_area instead of min_blob
            # Only consider components with area LESS than or equal to the threshold
            if stats[i, cv2.CC_STAT_AREA] <= min_blob:
                clean_mask[labels == i] = 1
                has_subtitle_pixels = True

        current_out_frame = np.zeros_like(roi)
        if has_subtitle_pixels:
            current_out_frame[clean_mask == 1] = (255, 255, 255)

        if processed_video_writer:
            try:
                processed_video_writer.write(current_out_frame)
            except Exception as e:
                logger.warning(f"Failed to write frame {frame_count} to processed video: {e}")

        is_different = frames_are_different_percentage(current_out_frame, last_out_frame, roi_area, change_percent_threshold, logger)

        if is_different:
            logger.info(f"Change detected via Pixel Percentage at frame {frame_count} ({current_frame_ms:.0f}ms)")
            keyframe_relative_path = None
            if last_out_frame is not None:
                segment_end_ms = current_frame_ms
                if segment_end_ms > current_segment_start_ms:
                    resized_keyframe = resize_frame(last_out_frame, target_frame_width)
                    keyframe_filename_base = f"keyframe_{keyframe_counter}.png"
                    keyframe_save_path = os.path.join(keyframes_dir, keyframe_filename_base)
                    keyframe_relative_path = os.path.join(KEYFRAMES_SUBDIR, keyframe_filename_base)
                    keyframe_counter += 1

                    try:
                        cv2.imwrite(keyframe_save_path, resized_keyframe)
                        logger.info(f"Saved individual keyframe image: {keyframe_save_path}")
                        individual_keyframes_info.append({
                            'timestamp_ms': current_segment_start_ms,
                            'absolute_path': keyframe_save_path,
                            'relative_path': keyframe_relative_path
                        })
                    except Exception as e:
                         logger.error(f"Failed to save keyframe image {keyframe_save_path}: {e}")
                         keyframe_relative_path = None

                    time_segments.append({
                        'start_ms': current_segment_start_ms,
                        'end_ms': segment_end_ms,
                        'keyframe_image': keyframe_relative_path
                    })
                    logger.info(f"Segment ended: {current_segment_start_ms:.0f}ms -> {segment_end_ms:.0f}ms (Individual Keyframe: {keyframe_relative_path})")
                else:
                    logger.info(f"Skipping segment ending at {segment_end_ms:.0f}ms (zero or negative duration).")

            current_segment_start_ms = current_frame_ms
            logger.info(f"Segment started: {current_segment_start_ms:.0f}ms")

        last_out_frame = current_out_frame.copy()

    pbar.close()

    # --- Finalize the last segment ---
    keyframe_relative_path = None
    if last_out_frame is not None:
        final_end_ms = duration_ms if duration_ms > current_segment_start_ms else current_frame_ms
        if final_end_ms > current_segment_start_ms:
            resized_keyframe = resize_frame(last_out_frame, target_frame_width)
            keyframe_filename_base = f"keyframe_{keyframe_counter}.png"
            keyframe_save_path = os.path.join(keyframes_dir, keyframe_filename_base)
            keyframe_relative_path = os.path.join(KEYFRAMES_SUBDIR, keyframe_filename_base)
            keyframe_counter += 1

            try:
                cv2.imwrite(keyframe_save_path, resized_keyframe)
                logger.info(f"Saved final individual keyframe image: {keyframe_save_path}")
                individual_keyframes_info.append({
                    'timestamp_ms': current_segment_start_ms,
                    'absolute_path': keyframe_save_path,
                    'relative_path': keyframe_relative_path
                })
            except Exception as e:
                 logger.error(f"Failed to save final keyframe image {keyframe_save_path}: {e}")
                 keyframe_relative_path = None

            time_segments.append({
                'start_ms': current_segment_start_ms,
                'end_ms': final_end_ms,
                'keyframe_image': keyframe_relative_path
            })
            logger.info(f"Final segment ended: {current_segment_start_ms:.0f}ms -> {final_end_ms:.0f}ms (Individual Keyframe: {keyframe_relative_path})")
        else:
            logger.info(f"Skipping final segment ending at {final_end_ms:.0f}ms (zero or negative duration).")

    cap.release()
    logger.info(f"Finished initial keyframe extraction. Found {len(individual_keyframes_info)} individual keyframes and {len(time_segments)} time segments.")

    # --- Create Composite Images ---
    composite_image_paths = []
    if individual_keyframes_info:
        logger.info(f"Creating composite keyframe images (batch size: {COMPOSITE_BATCH_SIZE}, layout: 1xN vertical)...")
        num_batches = math.ceil(len(individual_keyframes_info) / COMPOSITE_BATCH_SIZE)
        for i in range(num_batches):
            batch_start_index = i * COMPOSITE_BATCH_SIZE
            batch_end_index = batch_start_index + COMPOSITE_BATCH_SIZE
            batch_info = individual_keyframes_info[batch_start_index:batch_end_index]

            composite_filename = f"composite_{i}.png"
            composite_save_path = os.path.join(composite_keyframes_dir, composite_filename)

            # Use grid_cols=1 implicitly now for vertical stacking
            success, _ = create_composite_image(batch_info, composite_save_path, target_width=target_frame_width, logger=logger)
            if success:
                composite_image_paths.append(composite_save_path)
            else:
                logger.error(f"Failed to create or save composite image: {composite_save_path}")
        logger.info(f"Created {len(composite_image_paths)} composite images.")
    else:
        logger.warning("No individual keyframes found, skipping composite image creation.")


    if processed_video_writer:
        logger.info("Releasing processed video writer.")
        processed_video_writer.release()

    return individual_keyframes_info, time_segments, processed_video_path, composite_image_paths


# ---------- OCR and SRT Generation ----------
def generate_srt_from_keyframes(
    composite_image_paths: list,
    time_segments: list,
    api_keys: dict,
    prompt_file_path: str,
    process_dir: str,
    logger: logging.Logger,
    model: str = MODEL_GPT4O
) -> str:
    """
    Sends COMPOSITE keyframes and timing data to the selected AI model for OCR and SRT generation.
    Supports OpenAI GPT-4o and Claude Sonnet models.
    Saves inputs (prompt, timecodes) and output (SRT) to process_dir.
    Returns the path to the generated SRT file or raises an error.
    """
    is_claude = model.startswith("claude")
    api_provider = "Anthropic Claude" if is_claude else "OpenAI"
    logger.info(f"Starting SRT generation using {api_provider} ({model}) with composite images.")
    
    srt_path = os.path.join(process_dir, "output.srt")
    timecodes_path = os.path.join(process_dir, "timecodes.json")
    prompt_copy_path = os.path.join(process_dir, "prompt_used.txt")

    if not composite_image_paths or not time_segments:
        logger.warning("No composite keyframes or time segments available. Cannot generate SRT via API.")
        with open(srt_path, "w", encoding="utf-8") as f: f.write("")
        logger.info("Created empty SRT file as no content was detected for API call.")
        return srt_path

    # Check for required API key
    if is_claude and not api_keys.get('anthropic'):
        logger.error("Anthropic API key is required for Claude models but was not provided.")
        raise ValueError("Anthropic API key is required for Claude models.")
    elif not is_claude and not api_keys.get('openai'):
        logger.error("OpenAI API key is required for GPT-4o but was not provided.")
        raise ValueError("OpenAI API key is required for GPT-4o.")

    system_prompt = "Default prompt error: Could not load prompt_used.txt"
    try:
        with open(prompt_copy_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        logger.info(f"Loaded prompt for API call from {prompt_copy_path}")
    except Exception as e_load:
        logger.error(f"Failed to load prompt copy {prompt_copy_path}: {e_load}. Using default prompt.")

    # Common text content for both APIs
    text_content = f"Here is the JSON timing data for the original subtitle segments. Each segment includes the relative path to the *individual* keyframe image that represents the visual state at the start of that segment. You need to correlate the timestamps burned onto the sub-images within the composite images provided below with the 'start_ms' in this JSON data:\n```json\n{json.dumps(time_segments, indent=2)}\n```\nAnalyze the following composite keyframe images, extract text from the timestamped sub-images, correlate with the JSON, merge segments, and generate the SRT output."

    # Prepare image data
    valid_composites_sent = 0
    base64_images = []
    for composite_path in composite_image_paths:
        try:
            with open(composite_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                base64_images.append(base64_image)
                valid_composites_sent += 1
                logger.info(f"Adding composite image {os.path.basename(composite_path)} to API request.")
        except Exception as e:
            logger.error(f"Failed to read or encode composite image {composite_path}: {e}")

    if valid_composites_sent == 0:
        logger.warning("No valid composite images could be prepared for the API call. Skipping API request.")
        with open(srt_path, "w", encoding="utf-8") as f: f.write("")
        return srt_path

    try:
        srt_content = ""
        
        if is_claude:
            # Use Anthropic Claude API
            client = anthropic.Anthropic(api_key=api_keys['anthropic'])
            
            # Prepare Claude message with images
            message_content = [
                {"type": "text", "text": text_content}
            ]
            
            # Add images to the message
            for base64_img in base64_images:
                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_img
                    }
                })
            
            logger.info(f"Sending {valid_composites_sent} composite images and {len(time_segments)} segments to Anthropic Claude API.")
            
            response = client.messages.create(
                model=model,
                max_tokens=3000,
                temperature=0.7,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": message_content}
                ]
            )
            
            if response.content and len(response.content) > 0:
                # Extract text content from Claude response
                srt_content = "".join([block.text for block in response.content if block.type == "text"])
            else:
                logger.error("Claude API response did not contain expected content.")
                raise ValueError("Invalid response from Claude API.")
                
        else:
            # Use OpenAI API
            client = OpenAI(api_key=api_keys['openai'])
            
            # Prepare OpenAI message with images
            messages = [{"role": "system", "content": system_prompt}]
            user_content = [
                {"type": "text", "text": text_content}
            ]
            
            # Add images to the message
            for base64_img in base64_images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img}",
                        "detail": "low"
                    }
                })
            
            messages.append({"role": "user", "content": user_content})
            logger.info(f"Sending {valid_composites_sent} composite images and {len(time_segments)} segments to OpenAI API.")
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=3000,
                temperature=0.4
            )
            
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                srt_content = response.choices[0].message.content.strip()
            else:
                logger.error("OpenAI API response did not contain expected content.")
                finish_reason = response.choices[0].finish_reason if response.choices else 'N/A'
                logger.error(f"Finish Reason: {finish_reason}")
                raise ValueError(f"Invalid response from OpenAI API. Finish Reason: {finish_reason}")
        
        # Process and save the SRT content
        if "-->" not in srt_content:
            logger.warning(f"API response doesn't look like SRT content: {srt_content[:200]}...")
        else:
            logger.info(f"Successfully received plausible SRT content from {api_provider}.")
        
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        logger.info(f"SRT file saved to: {srt_path}")
        return srt_path

    except Exception as e:
        logger.error(f"Error during {api_provider} API call: {e}", exc_info=True)
        raise ConnectionError(f"Failed to get SRT from {api_provider}: {e}")


# ---------- Gradio UI and Main Logic ----------
def convert_srt_to_vtt(srt_path, logger):
    """Convert SRT subtitle file to VTT format for HTML5 video players."""
    try:
        vtt_path = srt_path.replace('.srt', '.vtt')
        subs = pysubs2.load(srt_path)
        subs.save(vtt_path)
        logger.info(f"Converted SRT to VTT: {vtt_path}")
        return vtt_path
    except Exception as e:
        logger.error(f"Error converting SRT to VTT: {e}")
        return None

def translate_srt(srt_path, target_language, api_keys, logger, model=MODEL_GPT4O):
    """
    Translate SRT subtitles to target language while preserving timing.
    Returns path to translated SRT file.
    """
    try:
        # Parse the input SRT file
        logger.info(f"Starting translation of SRT to {target_language} using {model}")

        # Check for required API key
        is_claude = model.startswith("claude")
        if is_claude and not api_keys.get('anthropic'):
            logger.error("Anthropic API key is required for translation with Claude models.")
            raise ValueError("Anthropic API key is required for translation.")
        elif not is_claude and not api_keys.get('openai'):
            logger.error("OpenAI API key is required for translation with GPT-4o.")
            raise ValueError("OpenAI API key is required for translation.")

        # Read the SRT content
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        if not srt_content or "-->" not in srt_content:
            logger.warning("No valid SRT content to translate")
            return None

        # Create translation prompt
        translation_prompt = f"""You are a professional subtitle translator. Your task is to translate the following SRT subtitle file to {target_language}.

CRITICAL REQUIREMENTS:
1. Preserve the EXACT SRT format including all timestamps
2. Keep the same subtitle numbering
3. Keep the exact same timing (the --> timestamps must not change)
4. Only translate the text content of each subtitle
5. Maintain natural flow and readability in the target language
6. If there are special characters or formatting, preserve them
7. Output ONLY the translated SRT content, no explanations or markdown
8. Make sure the translation fits naturally with the timing (not too long for short durations)

Here is the SRT content to translate:

{srt_content}

Remember: Output ONLY the translated SRT file content, nothing else."""

        try:
            translated_content = ""

            if is_claude:
                # Use Anthropic Claude API
                client = anthropic.Anthropic(api_key=api_keys['anthropic'])

                logger.info(f"Sending SRT for translation to Anthropic Claude ({target_language})")

                response = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": translation_prompt}
                    ]
                )

                if response.content and len(response.content) > 0:
                    translated_content = "".join([block.text for block in response.content if block.type == "text"])
                else:
                    logger.error("Claude API response did not contain translated content.")
                    raise ValueError("Invalid translation response from Claude API.")

            else:
                # Use OpenAI API
                client = OpenAI(api_key=api_keys['openai'])

                logger.info(f"Sending SRT for translation to OpenAI ({target_language})")

                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a professional subtitle translator. Translate subtitles while preserving exact SRT format and timing."},
                        {"role": "user", "content": translation_prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.3
                )

                if response.choices and response.choices[0].message and response.choices[0].message.content:
                    translated_content = response.choices[0].message.content.strip()
                else:
                    logger.error("OpenAI API response did not contain translated content.")
                    raise ValueError("Invalid translation response from OpenAI API.")

            # Validate translated content
            if "-->" not in translated_content:
                logger.error(f"Translation doesn't look like valid SRT: {translated_content[:200]}...")
                raise ValueError("Translation resulted in invalid SRT format")

            # Save translated SRT
            base_name = os.path.splitext(srt_path)[0]
            language_code = target_language.split(' - ')[0].lower().replace(' ', '_')
            translated_path = f"{base_name}_{language_code}.srt"

            with open(translated_path, 'w', encoding='utf-8') as f:
                f.write(translated_content)

            logger.info(f"Translated SRT saved to: {translated_path}")

            # Also create VTT version
            translated_vtt = convert_srt_to_vtt(translated_path, logger)

            return translated_path, translated_vtt

        except Exception as e:
            logger.error(f"Error during translation API call: {e}", exc_info=True)
            raise

    except Exception as e:
        logger.error(f"Error translating SRT: {e}")
        return None, None

def process_video_and_generate_srt(
    video_info,
    openai_api_key_input, anthropic_api_key_input,
    generate_srt_flag, ai_model_dropdown,
    save_processed_video_flag,
    white_lvl, tol, ratio_h, ratio_side, blob,
    threshold_value, # Percentage threshold input from Gradio
    frame_width,
    translate_flag=False,
    target_language=None
):
    # --- Setup ---
    if video_info is None:
        return None, None, None, "Error: Please upload a video file."

    video_path = video_info if isinstance(video_info, str) else video_info.name
    if not os.path.exists(video_path):
         video_path_rel = os.path.join(APP_DIR, os.path.basename(video_path))
         if not os.path.exists(video_path_rel):
              return None, None, None, f"Error: Input video path not found: {video_path} or {video_path_rel}"
         else: video_path = video_path_rel

    # Extract the actual model name from the dropdown value
    selected_model = ai_model_dropdown.split(" ")[0] if " " in ai_model_dropdown else ai_model_dropdown
    is_claude = selected_model.startswith("claude")
    
    # Set up API keys
    api_keys = get_api_keys()
    
    # Update API keys with user input if provided
    if openai_api_key_input:
        api_keys['openai'] = openai_api_key_input
        try:
            save_api_keys(openai_api_key=openai_api_key_input)
        except Exception as e:
            print(f"Warning: Failed to save OpenAI API key: {e}")
    
    if anthropic_api_key_input:
        api_keys['anthropic'] = anthropic_api_key_input
        try:
            save_api_keys(anthropic_api_key=anthropic_api_key_input)
        except Exception as e:
            print(f"Warning: Failed to save Anthropic API key: {e}")
    
    # Check if we have the required API key for the selected model
    if generate_srt_flag:
        if is_claude and not api_keys.get('anthropic'):
            return None, None, None, "Error: Anthropic API Key is required to generate SRT with Claude models."
        elif not is_claude and not api_keys.get('openai'):
            return None, None, None, "Error: OpenAI API Key is required to generate SRT with GPT-4o."

    process_dir, keyframes_output_dir, composite_keyframes_output_dir = create_process_output_dir()
    log_file = os.path.join(process_dir, "process.log")
    logger = setup_logger(log_file)
    status_updates = [f"Process started. Output directory: {process_dir}"]
    logger.info("="*20 + " New Process Started " + "="*20)
    logger.info(f"Process directory (abs): {os.path.abspath(process_dir)}")
    logger.info(f"Keyframes directory (abs): {os.path.abspath(keyframes_output_dir)}")
    logger.info(f"Composite Keyframes directory (abs): {os.path.abspath(composite_keyframes_output_dir)}")
    logger.info(f"Input video: {video_path}")
    logger.info(f"Generate SRT: {generate_srt_flag}")
    if generate_srt_flag:
        logger.info(f"AI Model: {selected_model}")
    logger.info(f"Save Processed Video: {save_processed_video_flag}")

    srt_file_path = None
    processed_video_path_out = None

    # --- Core Processing ---
    try:
        status_updates.append("Extracting individual keyframes and creating composites...")
        logger.info("Calling extract_subtitle_keyframes...")
        individual_keyframes, segments, processed_video_path_internal, composite_paths = extract_subtitle_keyframes(
            src=video_path,
            process_dir=process_dir,
            keyframes_dir=keyframes_output_dir,
            composite_keyframes_dir=composite_keyframes_output_dir,
            logger=logger,
            save_processed_video=save_processed_video_flag,
            white_lvl=int(white_lvl),
            tol=int(tol),
            keep_ratio_h=float(ratio_h),
            side_ratio=float(ratio_side),
            min_blob=int(blob),
            change_percent_threshold=float(threshold_value),
            target_frame_width=int(frame_width)
        )
        processed_video_path_out = processed_video_path_internal
        status_updates.append(f"Keyframe extraction complete. Found {len(individual_keyframes)} individual keyframes.")
        status_updates.append(f"Created {len(composite_paths)} composite keyframe images.")
        if processed_video_path_out:
             status_updates.append(f"Processed video saved to: {processed_video_path_out}")

        timecodes_path = os.path.join(process_dir, "timecodes.json")
        prompt_copy_path = os.path.join(process_dir, "prompt_used.txt")
        prompt_file_abs_path = os.path.join(APP_DIR, PROMPT_FILE_NAME)
        try:
            with open(timecodes_path, "w", encoding="utf-8") as f: json.dump(segments, f, indent=2)
            logger.info(f"Saved timecodes JSON to {timecodes_path}")
            status_updates.append(f"Timecodes saved to: {timecodes_path}")
        except Exception as e:
            logger.error(f"Failed to save timecodes JSON: {e}")
            status_updates.append(f"Error saving timecodes JSON: {e}")
        try:
            if os.path.exists(prompt_file_abs_path):
                shutil.copy2(prompt_file_abs_path, prompt_copy_path)
                logger.info(f"Copied prompt file to {prompt_copy_path}")
                status_updates.append(f"Prompt file copied to: {prompt_copy_path}")
            else:
                with open(prompt_copy_path, "w", encoding="utf-8") as f: f.write("Original prompt file not found.")
                logger.warning(f"Original prompt file not found at {prompt_file_abs_path}")
                status_updates.append(f"Original prompt file not found, placeholder saved.")
        except Exception as e:
            logger.error(f"Failed to copy prompt file: {e}")
            status_updates.append(f"Error copying prompt file: {e}")

        if generate_srt_flag:
            api_provider = "Anthropic Claude" if is_claude else "OpenAI"
            status_updates.append(f"Generating SRT via {api_provider} ({selected_model})...")
            logger.info("Calling generate_srt_from_keyframes...")
            srt_file_path = generate_srt_from_keyframes(
                composite_image_paths=composite_paths,
                time_segments=segments,
                api_keys=api_keys,
                prompt_file_path=prompt_file_abs_path,
                process_dir=process_dir,
                logger=logger,
                model=selected_model
            )
            srt_abs_path = os.path.join(process_dir, "output.srt")
            if srt_file_path and os.path.exists(srt_abs_path) and os.path.getsize(srt_abs_path) > 0:
                 status_updates.append(f"SRT generation complete. Saved to: {srt_file_path}")
            elif srt_file_path and os.path.exists(srt_abs_path):
                 status_updates.append("SRT generation resulted in an empty file (no subtitles detected by API?).")
            else:
                 status_updates.append("SRT generation failed (check logs).")
        else:
            status_updates.append("SRT generation skipped by user.")
            logger.info("SRT generation skipped by user.")

        # Create a video with subtitles if SRT was generated
        subtitled_video = None
        vtt_path = None
        translated_srt = None
        translated_vtt = None

        if generate_srt_flag and srt_file_path and os.path.exists(srt_file_path) and os.path.getsize(srt_file_path) > 0:
            # Convert SRT to VTT for the video player
            vtt_path = convert_srt_to_vtt(srt_file_path, logger)
            if vtt_path and os.path.exists(vtt_path):
                # Return a list with video path and subtitle path for the Gradio Video component
                subtitled_video = [video_path, vtt_path]
                status_updates.append("Video with subtitles ready for playback in the UI.")
            else:
                status_updates.append("Failed to convert SRT to VTT format. Make sure pysubs2 is installed.")

            # Handle translation if requested
            if translate_flag and target_language and target_language != "None":
                try:
                    status_updates.append(f"Translating subtitles to {target_language}...")
                    logger.info(f"Starting subtitle translation to {target_language}")
                    translated_srt, translated_vtt = translate_srt(
                        srt_file_path,
                        target_language,
                        api_keys,
                        logger,
                        model=selected_model
                    )
                    if translated_srt and os.path.exists(translated_srt):
                        status_updates.append(f"Translation complete. Saved to: {translated_srt}")
                    else:
                        status_updates.append("Translation failed. Check logs for details.")
                except Exception as e:
                    logger.error(f"Translation error: {e}")
                    status_updates.append(f"Translation error: {e}")

        final_status = "\n".join(status_updates) + "\n\nProcess completed successfully."
        logger.info("Process completed successfully.")
        return srt_file_path, log_file, processed_video_path_out, subtitled_video, vtt_path, translated_srt, translated_vtt, final_status

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        final_status = "\n".join(status_updates) + f"\n\nError during processing: {e}\nCheck log file for details: {log_file}"
        return None, log_file, None, None, None, None, None, final_status


# --- Gradio Interface Definition ---
with gr.Blocks(title="Video Subtitle Extractor (OCR)") as demo:
    gr.Markdown("# Video Subtitle Extractor using Frame Analysis and OCR")
    gr.Markdown(
        "Upload a video with hardcoded subtitles. The tool detects subtitle changes based on the percentage of changed pixels, "
        "extracts keyframes (saving individual frames and batched composites), and can optionally use AI Vision models to generate an SRT file or save the processed B&W video."
    )

    with gr.Accordion("API Keys (Required for SRT Generation)", open=False):
        api_keys = get_api_keys()
        openai_key_status = "OpenAI API Key not found in config.ini." if not api_keys['openai'] else "OpenAI API Key found in config.ini."
        anthropic_key_status = "Anthropic API Key not found in config.ini." if not api_keys['anthropic'] else "Anthropic API Key found in config.ini."
        
        gr.Markdown(f"**OpenAI API Key Status:** {openai_key_status}")
        openai_api_key_input = gr.Textbox(label="OpenAI API Key (for GPT-4o)", type="password", placeholder="Enter your OpenAI API Key here if not found/saved")
        
        gr.Markdown(f"**Anthropic API Key Status:** {anthropic_key_status}")
        anthropic_api_key_input = gr.Textbox(label="Anthropic API Key (for Claude models)", type="password", placeholder="Enter your Anthropic API Key here if not found/saved")

    with gr.Row():
        with gr.Column(scale=1):
            inp_video = gr.Video(label="Input Video")
            cb_generate_srt = gr.Checkbox(label="Generate SRT using AI", value=False)
            ai_model_dropdown = gr.Dropdown(
                label="AI Model for OCR",
                choices=[
                    MODEL_GPT4O + " (OpenAI)",
                    MODEL_CLAUDE_SONNET_35 + " (Claude 3.5 Sonnet)",
                    MODEL_CLAUDE_SONNET_37 + " (Claude 3.7 Sonnet)"
                ],
                value=MODEL_GPT4O + " (OpenAI)"
            )

            # Translation options
            with gr.Group():
                cb_translate = gr.Checkbox(label="Translate Subtitles", value=False)
                translate_language = gr.Dropdown(
                    label="Target Language for Translation",
                    choices=[
                        "None",
                        "English - English",
                        "Hebrew - עברית",
                        "Spanish - Español",
                        "French - Français",
                        "German - Deutsch",
                        "Italian - Italiano",
                        "Portuguese - Português",
                        "Russian - Русский",
                        "Chinese Simplified - 简体中文",
                        "Chinese Traditional - 繁體中文",
                        "Japanese - 日本語",
                        "Korean - 한국어",
                        "Arabic - العربية",
                        "Hindi - हिन्दी",
                        "Dutch - Nederlands",
                        "Turkish - Türkçe",
                        "Polish - Polski"
                    ],
                    value="None",
                    visible=False
                )

                # Show/hide language dropdown based on checkbox
                cb_translate.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[cb_translate],
                    outputs=[translate_language]
                )

            cb_save_processed_video = gr.Checkbox(label="Save Processed B&W Video", value=True)
            btn_run = gr.Button("Run Process", variant="primary")
        with gr.Column(scale=1):
             out_status = gr.Textbox(label="Status / Output Directory", lines=5, interactive=False)
             out_srt = gr.File(label="Output SRT File")
             out_log = gr.File(label="Process Log File")
             out_processed_video = gr.Video(label="Processed B&W Video Output", interactive=False)
             out_subtitled_video = gr.Video(label="Video with Subtitles")
             out_subtitles_vtt = gr.File(label="Subtitles File (VTT)", visible=True)
             # Translation output files
             out_translated_srt = gr.File(label="Translated SRT File", visible=False)
             out_translated_vtt = gr.File(label="Translated VTT File", visible=False)

    with gr.Accordion("Advanced Parameters", open=False):
         gr.Markdown("Adjust these parameters based on your video's subtitle characteristics.")
         with gr.Row():
              param_lvl = gr.Slider(minimum=180, maximum=250, value=201, step=1, label="Subtitle White Level ≥")
              param_tol = gr.Slider(minimum=0, maximum=150, value=100, step=1, label="Color Tolerance ≤")
              param_blob = gr.Slider(minimum=100, maximum=2500, value=2500, step=100, label="Max Subtitle Pixel Area")
         with gr.Row():
              param_ratio_h = gr.Slider(minimum=0.05, maximum=0.50, value=0.15, step=0.01, label="Subtitle Area Height %")
              param_ratio_side = gr.Slider(minimum=0.00, maximum=0.45, value=0.20, step=0.01, label="Crop % Each Side")
         with gr.Row():
              # Change Detection Threshold (Percentage)
              param_change_thresh_percent = gr.Slider(
                  minimum=0.1, maximum=10.0, value=0.7, step=0.1,
                  label="Change Detection Threshold (%)"
              )
              param_frame_width = gr.Slider(minimum=256, maximum=1024, value=704, step=64, label="Keyframe Width for OCR")


    # Update visibility of translation outputs based on checkbox
    def update_translation_outputs(translate_checked):
        return (
            gr.update(visible=translate_checked),  # out_translated_srt
            gr.update(visible=translate_checked)   # out_translated_vtt
        )

    cb_translate.change(
        fn=update_translation_outputs,
        inputs=[cb_translate],
        outputs=[out_translated_srt, out_translated_vtt]
    )

    btn_run.click(
        process_video_and_generate_srt,
        inputs=[
            inp_video, openai_api_key_input, anthropic_api_key_input,
            cb_generate_srt, ai_model_dropdown,
            cb_save_processed_video,
            param_lvl, param_tol, param_ratio_h, param_ratio_side, param_blob,
            param_change_thresh_percent, # Pass the percentage threshold
            param_frame_width,
            cb_translate,  # Add translation flag
            translate_language  # Add target language
        ],
        outputs=[out_srt, out_log, out_processed_video, out_subtitled_video, out_subtitles_vtt, out_translated_srt, out_translated_vtt, out_status]
    )

if __name__ == "__main__":
    output_dir_main = os.path.join(APP_DIR, "output")
    if not os.path.exists(output_dir_main): os.makedirs(output_dir_main)
    config_file_main = get_config_path()
    try:
        if not os.path.exists(config_file_main):
             save_api_key("")
             os.remove(config_file_main)
        else: _ = get_api_key()
    except Exception as e:
        print(f"Warning: Could not access or create {config_file_main}. API key saving might fail. Error: {e}")

    demo.launch()
