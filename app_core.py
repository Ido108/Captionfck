#!/usr/bin/env python3
# Core processing functions for subtitle extraction - NO UI CODE

import cv2
import numpy as np
import os
from tqdm import tqdm
import datetime
import configparser
import logging
import base64
import json
import shutil
import math
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
COMPOSITE_BATCH_SIZE = 6
COMPOSITE_BORDER_SIZE = 2

# --- AI Model Constants ---
MODEL_CLAUDE_45_SONNET = "claude-sonnet-4-5-20250929"
MODEL_CLAUDE_4_SONNET = "claude-sonnet-4-20250514"
MODEL_CLAUDE_37_SONNET = "claude-3-7-sonnet-20250219"
MODEL_GPT5_CHAT = "gpt-5-chat-latest"
MODEL_GPT41 = "gpt-4.1-2025-04-14"
MODEL_GPT41_MINI = "gpt-4.1-mini"
MODEL_GPT4O = "gpt-5"
MODEL_O4_MINI = "o4-mini"

# --- Configuration Handling ---
def get_config_path():
    return os.path.join(APP_DIR, CONFIG_FILE_NAME)

def get_api_keys():
    config_path = get_config_path()
    config = configparser.ConfigParser()
    keys = {'openai': None, 'anthropic': None}

    if os.path.exists(config_path):
        config.read(config_path)
        keys['openai'] = config.get('Credentials', 'openai_api_key', fallback=None)
        keys['anthropic'] = config.get('Credentials', 'anthropic_api_key', fallback=None)

    return keys

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
    base_output_dir = os.path.join(APP_DIR, base_output_dir_name)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    process_dir = os.path.join(base_output_dir, f"process_{timestamp}")
    keyframes_dir = os.path.join(process_dir, KEYFRAMES_SUBDIR)
    composite_keyframes_dir = os.path.join(process_dir, COMPOSITE_KEYFRAMES_SUBDIR)
    os.makedirs(keyframes_dir, exist_ok=True)
    os.makedirs(composite_keyframes_dir, exist_ok=True)
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
    timestamp_str = format_srt_timestamp(timestamp_ms)

    if keyframe_path:
        keyframe_name = os.path.basename(keyframe_path).split('.')[0]
        timestamp_str = f"{timestamp_str} - {keyframe_name}"

    h, w = image.shape[:2]
    extra_space = 30
    new_h = h + extra_space

    if len(image.shape) == 2 or image.shape[2] == 1:
        new_image = np.zeros((new_h, w), dtype=image.dtype)
        new_image[extra_space:, :] = image
        new_image = cv2.cvtColor(new_image, cv2.COLOR_GRAY2BGR)
    else:
        new_image = np.zeros((new_h, w, 3), dtype=image.dtype)
        new_image[extra_space:, :] = image

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)
    (text_width, text_height), baseline = cv2.getTextSize(timestamp_str, font, font_scale, thickness)

    org = ((w - text_width) // 2, (extra_space + text_height) // 2)

    bg_top_left = (org[0]-3, org[1]-text_height-3)
    bg_bottom_right = (org[0]+text_width+3, org[1]+baseline+3)
    bg_top_left = (max(0, bg_top_left[0]), max(0, bg_top_left[1]))

    cv2.rectangle(new_image, bg_top_left, bg_bottom_right, (0,0,0), -1)
    cv2.putText(new_image, timestamp_str, org, font, font_scale, color, thickness, cv2.LINE_AA)

    return new_image

def create_composite_image(batch_keyframes_info, composite_save_path, target_width=512, logger=None):
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

    final_images = []
    current_total_h_check = 0
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

    final_composite_h = current_total_h_check + max(0, len(final_images) - 1) * COMPOSITE_BORDER_SIZE
    composite_image = np.zeros((final_composite_h, max_w, 3), dtype=np.uint8)

    current_y = 0
    for i, img in enumerate(final_images):
        h = img.shape[0]
        if current_y + h > composite_image.shape[0]:
             if logger: logger.warning(f"Image {i} exceeds composite height, skipping.")
             continue
        composite_image[current_y:current_y+h, :] = img
        current_y += h
        if i < len(final_images) - 1:
            border_y_start = current_y
            border_y_end = current_y + COMPOSITE_BORDER_SIZE
            if border_y_end <= composite_image.shape[0]:
                 composite_image[border_y_start:border_y_end, :] = (255, 255, 255)
            current_y += COMPOSITE_BORDER_SIZE

    try:
        cv2.imwrite(composite_save_path, composite_image)
        if logger: logger.info(f"Saved composite keyframe image (1x{len(final_images)} with borders): {composite_save_path}")
        return True, processed_image_paths
    except Exception as e:
        if logger: logger.error(f"Failed to save composite keyframe image {composite_save_path}: {e}")
        return False, []

def frames_are_different_percentage(frame1, frame2, roi_area, threshold_percent=1.0, logger=None):
    if frame1 is None or frame2 is None: return frame1 is not frame2
    if frame1.shape != frame2.shape: return True

    if len(frame1.shape) == 3:
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        frame1_gray = frame1
        frame2_gray = frame2

    white_mask1 = (frame1_gray > 200).astype(np.uint8)
    white_mask2 = (frame2_gray > 200).astype(np.uint8)

    diff_mask = cv2.absdiff(white_mask1, white_mask2)
    diff_white_count = np.count_nonzero(diff_mask)
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
) -> tuple:
    keyframe_counter = 1

    logger.info(f"--- Starting Keyframe Extraction ---")
    logger.info(f"Source Video: {src}")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error("Failed to open video file.")
        raise IOError("bad video")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_ms = (total_frames / fps) * 1000 if total_frames > 0 and fps > 0 else 0
    logger.info(f"Video properties: {w}x{h} @ {fps:.2f} fps, {total_frames} frames")

    roi_h = int(h * keep_ratio_h)
    y0 = h - roi_h
    x0 = int(w * side_ratio)
    roi_w = w - 2 * x0
    if roi_w <= 0 or roi_h <= 0:
        logger.error(f"Invalid ROI dimensions ({roi_w}x{roi_h})")
        cap.release()
        raise ValueError("Invalid ROI dimensions")
    logger.info(f"Subtitle ROI: x={x0}, y={y0}, width={roi_w}, height={roi_h}")
    roi_area = roi_w * roi_h

    processed_video_writer = None
    processed_video_path = None
    if save_processed_video:
        processed_video_filename = "processed_video.mp4"
        processed_video_path = os.path.join(process_dir, processed_video_filename)
        try:
            processed_video_writer = writer(processed_video_path, fps, (roi_w, roi_h))
            logger.info(f"Initialized processed video writer")
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
            logger.info(f"Change detected at frame {frame_count}")
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
                        logger.info(f"Saved keyframe: {keyframe_save_path}")
                        individual_keyframes_info.append({
                            'timestamp_ms': current_segment_start_ms,
                            'absolute_path': keyframe_save_path,
                            'relative_path': keyframe_relative_path
                        })
                    except Exception as e:
                         logger.error(f"Failed to save keyframe: {e}")
                         keyframe_relative_path = None

                    time_segments.append({
                        'start_ms': current_segment_start_ms,
                        'end_ms': segment_end_ms,
                        'keyframe_image': keyframe_relative_path
                    })

            current_segment_start_ms = current_frame_ms

        last_out_frame = current_out_frame.copy()

    pbar.close()

    # Finalize last segment
    keyframe_relative_path = None
    if last_out_frame is not None:
        final_end_ms = duration_ms if duration_ms > current_segment_start_ms else current_frame_ms
        if final_end_ms > current_segment_start_ms:
            resized_keyframe = resize_frame(last_out_frame, target_frame_width)
            keyframe_filename_base = f"keyframe_{keyframe_counter}.png"
            keyframe_save_path = os.path.join(keyframes_dir, keyframe_filename_base)
            keyframe_relative_path = os.path.join(KEYFRAMES_SUBDIR, keyframe_filename_base)

            try:
                cv2.imwrite(keyframe_save_path, resized_keyframe)
                logger.info(f"Saved final keyframe")
                individual_keyframes_info.append({
                    'timestamp_ms': current_segment_start_ms,
                    'absolute_path': keyframe_save_path,
                    'relative_path': keyframe_relative_path
                })
            except Exception as e:
                 logger.error(f"Failed to save final keyframe: {e}")
                 keyframe_relative_path = None

            time_segments.append({
                'start_ms': current_segment_start_ms,
                'end_ms': final_end_ms,
                'keyframe_image': keyframe_relative_path
            })

    cap.release()
    logger.info(f"Finished keyframe extraction. Found {len(individual_keyframes_info)} keyframes")

    # Create Composite Images
    composite_image_paths = []
    if individual_keyframes_info:
        logger.info(f"Creating composite keyframe images...")
        num_batches = math.ceil(len(individual_keyframes_info) / COMPOSITE_BATCH_SIZE)
        for i in range(num_batches):
            batch_start_index = i * COMPOSITE_BATCH_SIZE
            batch_end_index = batch_start_index + COMPOSITE_BATCH_SIZE
            batch_info = individual_keyframes_info[batch_start_index:batch_end_index]

            composite_filename = f"composite_{i}.png"
            composite_save_path = os.path.join(composite_keyframes_dir, composite_filename)

            success, _ = create_composite_image(batch_info, composite_save_path, target_width=target_frame_width, logger=logger)
            if success:
                composite_image_paths.append(composite_save_path)

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
    is_claude = model.startswith("claude")
    api_provider = "Anthropic Claude" if is_claude else "OpenAI"
    logger.info(f"Starting SRT generation using {api_provider} ({model})")

    srt_path = os.path.join(process_dir, "output.srt")
    prompt_copy_path = os.path.join(process_dir, "prompt_used.txt")

    if not composite_image_paths or not time_segments:
        logger.warning("No keyframes available")
        with open(srt_path, "w", encoding="utf-8") as f: f.write("")
        return srt_path

    if is_claude and not api_keys.get('anthropic'):
        raise ValueError("Anthropic API key required")
    elif not is_claude and not api_keys.get('openai'):
        raise ValueError("OpenAI API key required")

    system_prompt = "Default prompt"
    try:
        with open(prompt_copy_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
    except:
        pass

    text_content = f"Here is the JSON timing data for the original subtitle segments. Each segment includes the relative path to the *individual* keyframe image that represents the visual state at the start of that segment. You need to correlate the timestamps burned onto the sub-images within the composite images provided below with the 'start_ms' in this JSON data:\n```json\n{json.dumps(time_segments, indent=2)}\n```\nAnalyze the following composite keyframe images, extract text from the timestamped sub-images, correlate with the JSON, merge segments, and generate the SRT output."

    valid_composites_sent = 0
    base64_images = []
    for composite_path in composite_image_paths:
        try:
            with open(composite_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                base64_images.append(base64_image)
                valid_composites_sent += 1
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")

    if valid_composites_sent == 0:
        with open(srt_path, "w", encoding="utf-8") as f: f.write("")
        return srt_path

    try:
        srt_content = ""

        if is_claude:
            client = anthropic.Anthropic(api_key=api_keys['anthropic'])

            message_content = [{"type": "text", "text": text_content}]

            for base64_img in base64_images:
                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64_img
                    }
                })

            response = client.messages.create(
                model=model,
                max_tokens=3000,
                temperature=0.7,
                system=system_prompt,
                messages=[{"role": "user", "content": message_content}]
            )

            if response.content and len(response.content) > 0:
                srt_content = "".join([block.text for block in response.content if block.type == "text"])
            else:
                raise ValueError("Invalid response from Claude")

        else:
            client = OpenAI(api_key=api_keys['openai'])

            messages = [{"role": "system", "content": system_prompt}]
            user_content = [{"type": "text", "text": text_content}]

            for base64_img in base64_images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img}",
                        "detail": "low"
                    }
                })

            messages.append({"role": "user", "content": user_content})

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=3000,
                temperature=0.4
            )

            if response.choices and response.choices[0].message and response.choices[0].message.content:
                srt_content = response.choices[0].message.content.strip()
            else:
                raise ValueError("Invalid response from OpenAI")

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"SRT file saved")
        return srt_path

    except Exception as e:
        logger.error(f"Error during API call: {e}", exc_info=True)
        raise

def convert_srt_to_vtt(srt_path, logger):
    try:
        vtt_path = srt_path.replace('.srt', '.vtt')
        subs = pysubs2.load(srt_path)
        subs.save(vtt_path)
        logger.info(f"Converted SRT to VTT")
        return vtt_path
    except Exception as e:
        logger.error(f"Error converting SRT to VTT: {e}")
        return None

def translate_srt(srt_path, target_language, api_keys, logger, model=MODEL_GPT4O):
    try:
        logger.info(f"Starting translation to {target_language}")

        is_claude = model.startswith("claude")
        if is_claude and not api_keys.get('anthropic'):
            raise ValueError("Anthropic API key required")
        elif not is_claude and not api_keys.get('openai'):
            raise ValueError("OpenAI API key required")

        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        if not srt_content or "-->" not in srt_content:
            logger.warning("No valid SRT content")
            return None, None

        translation_prompt = f"""Translate this SRT to {target_language}. Keep exact timing and format:

{srt_content}

Output ONLY the translated SRT."""

        translated_content = ""

        if is_claude:
            client = anthropic.Anthropic(api_key=api_keys['anthropic'])
            response = client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.3,
                messages=[{"role": "user", "content": translation_prompt}]
            )
            if response.content and len(response.content) > 0:
                translated_content = "".join([block.text for block in response.content if block.type == "text"])
        else:
            client = OpenAI(api_key=api_keys['openai'])
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Translate subtitles preserving SRT format"},
                    {"role": "user", "content": translation_prompt}
                ],
                max_tokens=4000,
                temperature=0.3
            )
            if response.choices and response.choices[0].message:
                translated_content = response.choices[0].message.content.strip()

        if "-->" not in translated_content:
            raise ValueError("Translation invalid")

        base_name = os.path.splitext(srt_path)[0]
        language_code = target_language.split(' - ')[0].lower().replace(' ', '_')
        translated_path = f"{base_name}_{language_code}.srt"

        with open(translated_path, 'w', encoding='utf-8') as f:
            f.write(translated_content)

        logger.info(f"Translated SRT saved")

        translated_vtt = convert_srt_to_vtt(translated_path, logger)

        return translated_path, translated_vtt

    except Exception as e:
        logger.error(f"Error translating: {e}")
        return None, None