#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image
import onnxruntime as ort

IMG_SIZE = 768
K_MAX = 10

def letterbox(img: Image.Image, size: int = IMG_SIZE):
    orig_w, orig_h = img.size
    scale = size / max(orig_w, orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, scale, pad_left, pad_top

def to_chw(img: Image.Image) -> np.ndarray:
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)

def load_support_images(support_dir: str):
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted(
        os.path.join(support_dir, f)
        for f in os.listdir(support_dir)
        if os.path.splitext(f)[1].lower() in exts
    )[:K_MAX]
    return [Image.open(p).convert("RGB") for p in paths]

def run_inference(session: ort.InferenceSession, support_dir: str, query_path: str) -> dict:
    support_imgs = load_support_images(support_dir)
    if not support_imgs:
        return {"error": f"no support images in {support_dir}"}

    query_img = Image.open(query_path).convert("RGB")
    orig_w, orig_h = query_img.size

    support_data = np.zeros((1, K_MAX, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    mask_data = np.zeros((1, K_MAX), dtype=np.float32)

    for i, img in enumerate(support_imgs):
        lb, _, _, _ = letterbox(img, IMG_SIZE)
        support_data[0, i] = to_chw(lb)
        mask_data[0, i] = 1.0

    query_lb, scale, pad_left, pad_top = letterbox(query_img, IMG_SIZE)
    query_data = to_chw(query_lb)[np.newaxis]

    best_box, best_score, bg_prob = session.run(
        ["best_box", "best_score", "bg_prob"],
        {
            "support_imgs": support_data,
            "support_mask": mask_data,
            "query_img": query_data,
        },
    )

    cx, cy, w, h = best_box[0].tolist()
    cx_lb = cx * IMG_SIZE
    cy_lb = cy * IMG_SIZE
    w_lb = w * IMG_SIZE
    h_lb = h * IMG_SIZE

    x1 = (cx_lb - w_lb / 2 - pad_left) / scale
    y1 = (cy_lb - h_lb / 2 - pad_top) / scale
    x2 = (cx_lb + w_lb / 2 - pad_left) / scale
    y2 = (cy_lb + h_lb / 2 - pad_top) / scale

    clamp = lambda v, lo, hi: max(lo, min(int(math.floor(v + 0.5)), hi))
    return {
        "x1": clamp(x1, 0, orig_w),
        "y1": clamp(y1, 0, orig_h),
        "x2": clamp(x2, 0, orig_w),
        "y2": clamp(y2, 0, orig_h),
        "score": float(best_score[0]),
        "bg_prob": float(bg_prob[0]),
    }

def build_session(model_path: str, device_id: int) -> ort.InferenceSession:
    providers = [
        ("CUDAExecutionProvider", {"device_id": device_id}),
        "CPUExecutionProvider",
    ]
    session = ort.InferenceSession(model_path, providers=providers)
    active = session.get_providers()
    print(json.dumps({"status": "ready", "providers": active}), flush=True)
    return session

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    try:
        session = build_session(args.model, args.device_id)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), flush=True)
        sys.exit(1)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            result = run_inference(session, req["support_dir"], req["query_image"])
        except Exception as exc:
            result = {"error": str(exc)}
        print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
