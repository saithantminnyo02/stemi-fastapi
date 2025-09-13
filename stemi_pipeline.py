import os, sys, json, base64, cv2, numpy as np, requests

# ==== CONFIG: fill these ====
RF_API_KEY        = "iGcuMhOntpBUnb5vE0BP"
MODEL_A_PRIME_ID  = "stemi_binary_detection-wojso/3"   # A′: STEMI vs Non-STEMI
MODEL_A_ID        = "stemi-detection-9v28h/3"   # A:   {STE-C, STE-U, Normal, Abnormal, STE-Mimic}
MODEL_B_ID        = "stemi_artery-xjdgh/1"   # B: LAD / LCX / RCA classifier
# Thresholds (start here; tweak after a few cases)
THR_STEMI      = 0.55   # binary gate (0.45–0.60)
THR_CLEAR      = 0.60   # clear STEMI in A
THR_SUM_AMBIG  = 0.75   # ambiguous if STE-C+STE-U above this
DEBUG = False
# ============================

def _pad_resize(gray, W=768, H=512):
    h, w = gray.shape[:2]
    scale = min(W/w, H/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W), 255, dtype=np.uint8)
    y0, x0 = (H - nh)//2, (W - nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if h > w:  # normalize: portrait -> landscape
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(2.0, (8,8))
    gray = clahe.apply(gray)
    return _pad_resize(gray, 768, 512)

def preprocess_keep_as_is(path, target_w=768, target_h=512):
    """No rotation/CLAHE; grayscale + letterbox only (to match how you may have tested Model B in the UI)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _pad_resize(gray, target_w, target_h)

def pick_top(probs: dict):
    return (None, 0.0) if not probs else max(probs.items(), key=lambda kv: kv[1])

def rf_classify_bytes(model_id, jpg_bytes):
    url = f"https://detect.roboflow.com/{model_id}?api_key={RF_API_KEY}&confidence=0"
    b64 = base64.b64encode(jpg_bytes).decode("utf-8")
    r = requests.post(url, data=b64, headers={"Content-Type":"application/x-www-form-urlencoded"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if DEBUG: print("[RF]", json.dumps(data)[:400])
    probs = {p["class"]: float(p.get("confidence",0.0)) for p in data.get("predictions", [])}
    if not probs and "top" in data:  # fallback shape
        probs[data["top"]["class"]] = float(data["top"].get("confidence",0.0))
    return probs

def tta_probs(model_id, gray):
    views = [gray]
    H,W = gray.shape[:2]
    for a in (-4,4):
        M = cv2.getRotationMatrix2D((W/2,H/2), a, 1.0)
        views.append(cv2.warpAffine(gray, M, (W,H), borderValue=255))
    for g in (0.9,1.1):
        views.append(np.clip(gray.astype(np.float32)*g, 0,255).astype(np.uint8))

    acc = {}
    for v in views:
        _, buf = cv2.imencode(".jpg", v, [int(cv2.IMWRITE_JPEG_QUALITY),95])
        p = rf_classify_bytes(model_id, buf.tobytes())
        for k,val in p.items():
            acc[k] = acc.get(k,0.0) + val
    n = len(views)
    for k in acc: acc[k] /= n
    return acc

def decide_hierarchy(probs_bin, probs_5, probs_artery=None):
    # --- A′ gate ---
    p_stemi = probs_bin.get("STEMI", probs_bin.get("stemi", 0.0))
    if p_stemi < THR_STEMI:
        # Try to report strongest Non-STEMI subclass if available
        non = {k:v for k,v in probs_5.items() if k in ["Normal","Abnormal","STE-Mimic"]}
        top_non = max(non, key=non.get) if non else "Non-STEMI"
        return f"Non-STEMI ({top_non})", {"P_STEMI": round(p_stemi,3), "top_non": top_non,
                                        "p_top_non": round(non.get(top_non,0.0),3) if non else 0.0}

    # --- A refine (only STEMI-relevant outputs) ---
    p_c = probs_5.get("STE-C", probs_5.get("ste-c", 0.0))
    p_u = probs_5.get("STE-U", probs_5.get("ste-u", 0.0))

    if max(p_c, p_u) >= THR_CLEAR:
        subtype = "STE-C" if p_c >= p_u else "STE-U"
        details = {"p_STE-C": round(p_c,3), "p_STE-U": round(p_u,3)}
        if probs_artery:
            # Raw artery pick
            artery = max(probs_artery, key=probs_artery.get)
            pa = probs_artery
            details["artery_raw"] = artery
            details["artery_probs"] = {k: round(v,3) for k,v in pa.items()}
            # Consistency rule: STE-C (anterior) rarely pairs with RCA. Require strong margin to keep RCA
            if subtype == "STE-C" and ("RCA" in pa and "LAD" in pa):
                if pa["RCA"] < pa.get("LAD", 0.0) + 0.15:
                    artery = "LAD"
            details["artery"] = artery
            return f"STEMI: Clear ({subtype}, {artery})", details
        return f"STEMI: Clear ({subtype})", details

    if (p_c + p_u) >= THR_SUM_AMBIG:
        return "STEMI: Ambiguous", {"sum_STE": round(p_c+p_u,3), "p_STE-C": round(p_c,3), "p_STE-U": round(p_u,3)}

    non = {k:v for k,v in probs_5.items() if k in ["Normal","Abnormal","STE-Mimic"]}
    top_non = max(non, key=non.get) if non else "Non-STEMI"
    return f"Non-STEMI ({top_non})", {
        "reason": "STEMI weak after refine",
        "p_STE-C": round(p_c,3),
        "p_STE-U": round(p_u,3),
        "P_STEMI_bin": round(p_stemi,3),
        "top_non": top_non,
        "p_top_non": round(non.get(top_non,0.0),3) if non else 0.0
    }

def run(image_path):
    gray = preprocess(image_path)
    # A′
    probs_bin = tta_probs(MODEL_A_PRIME_ID, gray)
    # A
    probs_5 = tta_probs(MODEL_A_ID, gray)
    probs_artery = None
    # Only run artery classifier when the binary gate passes and subtype is confidently clear
    p_stemi_bin = probs_bin.get("STEMI", probs_bin.get("stemi", 0.0))
    if p_stemi_bin >= THR_STEMI:
        p_c = probs_5.get("STE-C", probs_5.get("ste-c", 0.0))
        p_u = probs_5.get("STE-U", probs_5.get("ste-u", 0.0))
        if max(p_c, p_u) >= THR_CLEAR:
            # --- Dual-view inference for Model B (artery) ---
            # View A: standard preprocessed image used for A′/A
            probsA = tta_probs(MODEL_B_ID, gray)
            # View B: original-aspect (no rotation/CLAHE), only letterboxed
            gray_orig = preprocess_keep_as_is(image_path, 768, 512)
            probsB = tta_probs(MODEL_B_ID, gray_orig)
            # Pick the view with higher top-1 confidence
            (topA, confA) = pick_top(probsA)
            (topB, confB) = pick_top(probsB)
            probs_artery = probsA if confA >= confB else probsB
            if DEBUG:
                print(f"[Artery dual-view] A: {topA}={confA:.3f} | B: {topB}={confB:.3f}")
    decision, details = decide_hierarchy(probs_bin, probs_5, probs_artery)
    out = {
        "image": os.path.basename(image_path),
        "decision": decision,
        "binary_probs": {k: round(v,3) for k,v in probs_bin.items()},
        "five_class_probs": {k: round(v,3) for k,v in probs_5.items()},
        "artery_probs": {k: round(v,3) for k,v in probs_artery.items()} if probs_artery else None,
        "details": details,
    }
    print(json.dumps(out, indent=2))

# ---- Add this helper so we can call from an API ----
def classify_path(image_path: str):
    gray = preprocess(image_path)
    # A′
    probs_bin = tta_probs(MODEL_A_PRIME_ID, gray)
    # A
    probs_5 = tta_probs(MODEL_A_ID, gray)
    probs_artery = None

    # Only run artery classifier when the binary gate passes and subtype is confidently clear
    p_stemi_bin = probs_bin.get("STEMI", probs_bin.get("stemi", 0.0))
    if p_stemi_bin >= THR_STEMI:
        p_c = probs_5.get("STE-C", probs_5.get("ste-c", 0.0))
        p_u = probs_5.get("STE-U", probs_5.get("ste-u", 0.0))
        if max(p_c, p_u) >= THR_CLEAR:
            # Dual-view inference for Model B (artery)
            probsA = tta_probs(MODEL_B_ID, gray)
            gray_orig = preprocess_keep_as_is(image_path, 768, 512)
            probsB = tta_probs(MODEL_B_ID, gray_orig)
            (topA, confA) = pick_top(probsA)
            (topB, confB) = pick_top(probsB)
            probs_artery = probsA if confA >= confB else probsB

    decision, details = decide_hierarchy(probs_bin, probs_5, probs_artery)
    return {
        "image": os.path.basename(image_path),
        "decision": decision,
        "binary_probs": {k: round(v,3) for k,v in probs_bin.items()},
        "five_class_probs": {k: round(v,3) for k,v in probs_5.items()},
        "artery_probs": {k: round(v,3) for k,v in probs_artery.items()} if probs_artery else None,
        "details": details,
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stemi_pipeline.py /path/to/ecg.jpg")
        sys.exit(1)
    run(sys.argv[1])