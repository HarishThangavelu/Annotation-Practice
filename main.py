import json
import base64
import io
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model once
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/nuclio/best.pt")
model = YOLO(MODEL_PATH)

def infer(context, event):
    # --- 1. BROWSER & HEALTH CHECK GUARD ---
    if event.method == 'GET' or not event.body:
        return context.Response(
            body="Hey Harish! Function is UP and Running!",
            status_code=200
        )

    context.logger.info(f"Processing request. Method: {event.method}")
    
    # --- 2. DATA EXTRACTION ---
    image_data = event.body
    
    if isinstance(image_data, dict):
        if 'image' in image_data:
            image_data = image_data['image']
        else:
            return context.Response(body="Invalid JSON: missing 'image' key", status_code=400)

    elif isinstance(image_data, bytes) and image_data.startswith(b'{'):
        try:
            payload = json.loads(image_data)
            if 'image' in payload:
                image_data = payload['image']
        except:
            pass 

    # --- 3. IMAGE DECODING ---
    try:
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        context.logger.warn(f"Failed to open image: {e}")
        return context.Response(body=f"Error: Not a valid image. {e}", status_code=400)

    # --- 4. INFERENCE ---
    try:
        image_np = np.asarray(image)
        results = model.predict(image_np, verbose=False)[0]

        found_objects = []
        
        # --- CRITICAL FIX FOR OBB MODELS ---
        # Check if the model returned OBB (Oriented) or standard Boxes
        if results.obb is not None:
            # It is an OBB model! Use the .obb property
            # .xyxy gives the horizontal box wrapping the rotated object (compatible with CVAT)
            detections = results.obb
        else:
            # It is a standard model
            detections = results.boxes

        for box in detections:
            # Get the wrapping rectangle (xyxy) whether it's OBB or Normal
            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].tolist()
            
            label_index = int(box.cls.item())
            conf = float(box.conf.item())
            label_str = str(results.names[label_index])

            found_objects.append({
                "confidence": conf,
                "label": label_str,
                "type": "rectangle",
                "points": [x1, y1, x2, y2]
            })
            
        context.logger.info(f"Success. Found {len(found_objects)} objects.")

        return context.Response(
            body=json.dumps(found_objects),  
            headers={},
            content_type="application/json",
            status_code=200
        )
    except Exception as e:
        context.logger.error(f"Prediction failed: {e}")
        return context.Response(body=str(e), status_code=500)