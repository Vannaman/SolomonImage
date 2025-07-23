import os
import io
import base64
import json
import random
import colorsys
from flask import Flask, render_template_string, request, jsonify 

import cv2
import numpy as np
from PIL import Image # Used for image manipulation
import requests # For connecting to the external AI server

# Disable SSL warnings for local development/testing with requests.
# This is crucial for connecting to servers with self-signed or invalid SSL certificates.
requests.packages.urllib3.disable_warnings()

# --- Flask App Setup ---
app = Flask(__name__)

# --- Functions from exampleCode.py (Adapted for Web Output) ---

def random_colors(N, bright=True):
    """
    Generates random colors.
    Ensures N is at least 1 to prevent ZeroDivisionError.
    """
    N = max(1, N)
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def getClassName(project_config):
    """
    Extracts class names from a project configuration dictionary.
    Now correctly accesses 'class_name' nested within 'model'.
    Ensures a default list if no class_name is found or accessible.
    """
    class_names = []
    # Check if 'model' key exists and is a dictionary
    if 'model' in project_config and isinstance(project_config['model'], dict):
        model_data = project_config['model']
        # Check if 'class_name' key exists within 'model_data' and is a list
        if 'class_name' in model_data and isinstance(model_data['class_name'], list):
            for item in model_data['class_name']:
                if isinstance(item, dict) and 'name' in item:
                    class_names.append(item['name'])
    return class_names if class_names else ["Object 1", "Object 2", "Object 3", "Object 4", "Object 5"]

def render_masks_from_server_response(result_data, image_shape):
    """
    Decodes the unique mask format from your AI server response,
    mimicking the 'rendering' function from your exampleCode.py.
    Expected format: a single byte string with masks separated by a delimiter.
    Args:
        result_data: The 'data' dictionary from the AI server's inference response (already the nested 'data').
        image_shape: The (height, width) of the original image for dummy masks.
    Returns:
        A list of decoded numpy mask arrays (binary 0/255) or empty list if no masks.
    """
    masks_raw = result_data.get('masks', '')
    masks_shape_list = result_data.get('masks_shape_list', [])
    
    mask_list = []
    
    try:
        if isinstance(masks_raw, str):
            masks_bytes = masks_raw.encode('utf-8')
        elif isinstance(masks_raw, bytes):
            masks_bytes = masks_raw
        else:
            print("Warning: Masks data is not string or bytes. Cannot decode.")
            return [np.zeros(image_shape, dtype=np.uint8)] * len(result_data.get('rois', []))

        # The custom delimiter from your exampleCode.py
        custom_delimiter = b"$$$$$$$$$$$$$finalResultMASK&&&&&&&&&&&&&"
        all_masks_parts = masks_bytes.split(custom_delimiter)

        # Filter out any empty byte strings that might result from splitting
        valid_mask_parts = [part for part in all_masks_parts if part]

        # Iterate based on the number of shapes provided, which should match detected masks
        for i in range(len(masks_shape_list)):
            if i < len(valid_mask_parts):
                mask_byte_data = valid_mask_parts[i]
                mask_h, mask_w = masks_shape_list[i]
                
                mask = np.frombuffer(mask_byte_data, dtype=np.uint8)
                
                # Reshape and ensure it's binary (0 or 255)
                # Check for size mismatch before reshape to prevent errors
                if mask.size != (mask_h * mask_w):
                    print(f"WARNING: Mask part {i} size mismatch (expected {mask_h * mask_w}, got {mask.size}). Creating blank mask.")
                    mask_list.append(np.zeros(image_shape, dtype=np.uint8))
                    continue

                mask = mask.reshape(mask_h, mask_w)
                mask[mask > 0] = 255 # Ensure it's binary 0/255
                mask_list.append(mask)
            else:
                print(f"INFO: Missing mask data for shape {i}. Appending blank mask.")
                mask_list.append(np.zeros(image_shape, dtype=np.uint8)) # Append blank mask if part is missing
        
        if not mask_list and masks_shape_list and result_data.get('rois'):
            # If masks_shape_list has entries but mask_list is still empty, and there are ROIs,
            # it means there was an issue in decoding. Provide blank masks matching ROI count.
            print("INFO: Masks list is empty after decoding, but ROIs exist. Creating blank masks matching ROI count.")
            return [np.zeros(image_shape, dtype=np.uint8)] * len(result_data['rois'])

        return mask_list
    except Exception as e:
        print(f"ERROR: Overall mask rendering/decoding failed: {e}. Returning None or blank masks.")
        # If any error, return None or an empty list, let the caller handle it or provide dummy masks
        return None # Let display_instances_cv2_for_web handle None, which appends dummy masks


def display_instances_cv2_for_web(image_np_bgr, boxes, masks, kpts, class_ids, scores,
                                  class_name=None, colors=None, show_scores=False, show_mask=True,
                                  show_bbox=True, show_kpts=True, draw_boundary=True):
    """
    Draws bounding boxes, masks, keypoints, and labels on an image using OpenCV.
    Returns a base64 encoded image instead of saving locally.
    Assumes image_np_bgr input is already in BGR format for OpenCV.
    """
    # --- NEW DEBUG PRINT ---
    print(f"DEBUG: display_instances_cv2_for_web received boxes (raw): {boxes}")
    print(f"DEBUG: display_instances_cv2_for_web received boxes.shape: {boxes.shape if hasattr(boxes, 'shape') else 'N/A'}")
    N = boxes.shape[0] if hasattr(boxes, 'shape') and boxes.shape else 0
    print(f"DEBUG: display_instances_cv2_for_web calculated N: {N}")
    # --- END NEW DEBUG PRINT ---

    # Handle cases where no objects are detected or no image to draw on
    if image_np_bgr is None or image_np_bgr.size == 0:
        print("display_instances_cv2_for_web: No image data provided. Returning blank image.")
        pil_img = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8)) # Blank black image
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    if N == 0:
        print("display_instances_cv2_for_web: No objects detected for drawing. Returning original image.")
        pil_img = Image.fromarray(cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB))
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    masked_image = image_np_bgr.astype(np.uint8).copy() # Ensure working on a copy and uint8
    overlay = masked_image.copy() # For translucent masks

    if colors is None or not colors:
        # Generate enough colors for all possible class IDs or a default of 10
        max_class_id = -1
        if class_ids is not None and len(class_ids) > 0:
            max_class_id = int(np.max(class_ids)) # Use np.max for numpy array
        num_colors_to_generate = max(10, len(class_name) if class_name else 0, max_class_id + 1)
        colors = random_colors(num_colors_to_generate)

    for i in range(N):
        color_idx = class_ids[i] % len(colors) if class_ids is not None and len(colors) > 0 else 0
        color_rgb = colors[color_idx]
        color_bgr = tuple([int(c * 255) for c in color_rgb[::-1]]) # Convert to BGR for OpenCV

        # Bounding box
        if show_bbox and boxes is not None and i < len(boxes):
            x1, y1, x2, y2 = boxes[i].astype(int) # Ensure integer coordinates
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), color_bgr, 3)

        # Mask
        if show_mask and masks is not None and i < len(masks) and masks[i] is not None:
            mask_single_obj = masks[i]
            if not isinstance(mask_single_obj, np.ndarray):
                try:
                    mask_single_obj = np.array(mask_single_obj, dtype=np.uint8)
                except Exception as e:
                    print(f"Warning: Could not convert mask {i} to numpy array in drawing: {e}. Skipping mask.")
                    mask_single_obj = np.zeros(masked_image.shape[:2], dtype=np.uint8) # Create a blank mask
            
            if mask_single_obj.ndim == 2:
                # Ensure mask is binary (0 or 255)
                mask_single_obj[mask_single_obj > 0] = 255
            else:
                print(f"Warning: Mask {i} is not 2D or convertible ({mask_single_obj.shape}). Skipping mask.")
                mask_single_obj = np.zeros(masked_image.shape[:2], dtype=np.uint8) # Default to blank if conversion fails

            if np.any(mask_single_obj): # Only draw if there's an actual mask to draw
                # Ensure mask has same dimensions as image if not already
                if mask_single_obj.shape != masked_image.shape[:2]:
                    # Resize mask to image dimensions if necessary (e.g., if masks are smaller than image)
                    mask_single_obj = cv2.resize(mask_single_obj, 
                                                (masked_image.shape[1], masked_image.shape[0]), 
                                                interpolation=cv2.INTER_NEAREST)

                contours, _ = cv2.findContours(mask_single_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if draw_boundary:
                    cv2.drawContours(masked_image, contours, -1, color_bgr, 2)
                else:
                    cv2.fillPoly(overlay, contours, color_bgr)
        elif show_mask: 
            print(f"INFO: display_instances_cv2_for_web: No valid mask data found for object {i} from input. Not drawing mask.")

        # Keypoints
        if show_kpts and kpts is not None and i < len(kpts):
            for kpt_val in kpts[i]:
                if len(kpt_val) >= 2:
                    kpt_x, kpt_y = int(kpt_val[0]), int(kpt_val[1])
                    cv2.circle(masked_image, (kpt_x, kpt_y), 3, (0, 0, 255), -1)

        # Label and Score
        label = ""
        # CRITICAL FIX: Ensure class_ids[i] is a valid index for class_name list
        if class_name is not None and class_ids is not None and i < len(class_ids) and class_ids[i] < len(class_name):
            label = class_name[class_ids[i]]
        else:
            # Fallback label if class_id is out of bounds for class_name list
            label = f"Object {class_ids[i]}" if class_ids is not None and i < len(class_ids) else "Unknown"

        if show_scores and scores is not None and i < len(scores):
            label = f"{label}: {scores[i]:.2f}"
        
        if show_bbox and boxes is not None and i < len(boxes):
            x1, y1, _, _ = boxes[i].astype(int)
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20 
            cv2.putText(masked_image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2, cv2.LINE_AA)

    if show_mask and not draw_boundary:
        cv2.addWeighted(overlay, 0.5, masked_image, 0.5, 0, masked_image)

    # Convert the final OpenCV BGR image to RGB for PIL to save as JPEG
    pil_img = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --- Server Client Class (Calls to external AI server) ---
class ExternalAIServerClient:
    """
    Handles direct communication with the external AI inference server.
    This class wraps requests to add common error handling and SSL bypass.
    """
    def __init__(self, server_ip, port='5000', use_https=False):
        protocol = "https" if use_https else "http"
        self.server_base_url = f'{protocol}://{server_ip}:{port}/sol_server'
        print(f"ExternalAIServerClient initialized to: {self.server_base_url}")
        # Disable specific urllib3 warnings if verify=False is used widely
        requests.packages.urllib3.disable_warnings()

    def _make_request(self, method, endpoint, **kwargs):
        """Helper to make requests with verify=False and common error handling."""
        url = f'{self.server_base_url}{endpoint}'
        print(f"DEBUG: Making {method} request to external AI server: {url}") # Debug print
        try:
            kwargs['verify'] = False # Bypass SSL certificate verification
            response = requests.request(method, url, **kwargs)
            
            print(f"DEBUG: Server Response Status Code: {response.status_code}") # NEW DEBUG
            print(f"DEBUG: Server Response Headers: {json.dumps(dict(response.headers), indent=2)}") # NEW DEBUG
            print(f"DEBUG: Raw Server Response Text: '{response.text[:200]}...'") # Print first 200 chars for brevity

            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            # Try to parse JSON. This might fail if response.text is empty or not JSON.
            json_response = response.json()
            print(f"DEBUG: Parsed JSON response from external AI server ({url}): {json.dumps(json_response, indent=2)}") # Debug print
            return {"success": True, "data": json_response}
        except requests.exceptions.SSLError as e:
            print(f"SSL Certificate Error for {url}: {e}")
            return {"success": False, "message": f"SSL Error: {e}. Check server's certificate."}
        except requests.exceptions.ConnectionError as e:
            print(f"Connection Error for {url}: {e}")
            return {"success": False, "message": f"Connection Error: {e}. Is server running and accessible?"}
        except requests.exceptions.Timeout as e:
            print(f"Timeout Error for {url}: {e}")
            return {"success": False, "message": f"Timeout: {e}. Server took too long to respond."}
        except json.JSONDecodeError as e: # This will catch if response.json() fails
            print(f"JSON Decode Error for {url}: {e}. Server response was not valid JSON or was empty.")
            # We already printed response.text above, which is crucial here
            return {"success": False, "message": f"Invalid JSON response from server: {e}. Raw: '{response.text}'"}
        except requests.exceptions.RequestException as e: # Catch any other requests errors
            print(f"General Request Error for {url}: {e}")
            response_content = "N/A"
            if hasattr(e, 'response') and e.response is not None:
                response_content = e.response.text
            print(f"Server Response: {response_content}")
            return {"success": False, "message": f"API Error: {e}. Server Response: {response_content}"}


    def get_all_projects(self):
        """Retrieves all projects from the server."""
        response = self._make_request('GET', '/project/all_projects')
        return response

    def get_single_project(self, project_id):
        """
        Retrieves details for a single project by ID.
        This function now returns the raw wrapped response from _make_request.
        The Flask route will handle unpacking it.
        """
        response = self._make_request('GET', f'/project/{project_id}')
        return response # Returns {"success": bool, "data": AI_SERVER_JSON_RESPONSE}

    def RegisterJobs(self, config):
        """Registers a new inference job."""
        return self._make_request('POST', '/job/trigger', json=config)

    def inference(self, img_bytes, tool_name, model_id, job_id, state_id=None):
        """Sends an image for inference and retrieves results."""
        headers = {
            'threshold': str(0.1), # Explicitly cast to str
            'Model-Id': str(model_id),
            'Tool-Name': tool_name,
            'Job-Id' : str(job_id),
            'State-Id' : str(state_id), # CRITICAL FIX: Always include, even if state_id is None
            'split_text': str(0) # Explicitly cast to str
        }

        files = {'files': ('image.jpg', img_bytes, 'image/jpeg')} # Using a default filename and mimetype

        return self._make_request('POST', '/inference', files=files, headers=headers, timeout=120)


# --- Flask Routes (Your Local Web App's Backend) ---

# Global client instance for the external AI server (initialized once)
external_ai_client = None

@app.route('/')
def index():
    """Serves the main HTML application."""
    # This HTML includes Tailwind CSS from CDN and all necessary JavaScript
    # JavaScript fetches data from Flask's /api/* endpoints
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/init', methods=['POST'])
def init_client():
    """Initializes the external AI client with user-provided server details."""
    global external_ai_client
    data = request.json
    server_ip = data.get('serverIp')
    server_port = data.get('serverPort')
    use_https = data.get('useHttps')

    if not server_ip or not server_port:
        return jsonify({"success": False, "message": "Server IP and Port are required."}), 400

    external_ai_client = ExternalAIServerClient(server_ip, server_port, use_https)
    return jsonify({"success": True, "message": "Client initialized."})

@app.route('/api/projects', methods=['GET'])
def get_projects():
    """Fetches projects from the external AI server."""
    if not external_ai_client:
        return jsonify({"success": False, "message": "Client not initialized. Set server details first."}), 400
    
    response = external_ai_client.get_all_projects()
    # Ensure Flask always returns the 'success' key, consistent with frontend expectation
    return jsonify(response) # Return the entire response object including 'success' and 'data'

@app.route('/api/project/<project_id>', methods=['GET'])
def get_project_config(project_id):
    """
    Fetches a single project's configuration from the external AI server.
    This route now correctly unwraps the nested 'data' from the AI server's response.
    """
    if not external_ai_client:
        return jsonify({"success": False, "message": "Client not initialized."}), 400
    
    # This 'wrapped_response_from_external_client' is like {"success": True, "data": AI_SERVER_FULL_RESPONSE}
    wrapped_response_from_external_client = external_ai_client.get_single_project(project_id)

    if wrapped_response_from_external_client['success']:
        # AI_SERVER_FULL_RESPONSE is like {"data": [actual_configs], "message": ..., "result": true}
        # We need to extract the 'data' key from AI_SERVER_FULL_RESPONSE.
        ai_server_data_payload = wrapped_response_from_external_client['data']
        
        # Now, actual_project_configs_list is the list of actual project configurations
        actual_project_configs_list = ai_server_data_payload.get('data', [])

        # The frontend expects 'result.data' to be an array directly,
        # and it then takes result.data[0] for the specific config.
        return jsonify({"success": True, "data": actual_project_configs_list})
    else:
        # Pass the original error message from the external client to the frontend
        return jsonify({"success": False, "message": wrapped_response_from_external_client['message']}), 500

@app.route('/api/inference', methods=['POST'])
def run_inference():
    """Handles image upload and inference request to the external AI server."""
    if not external_ai_client:
        return jsonify({"success": False, "message": "Client not initialized."}), 400

    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image file provided."}), 400

    image_file = request.files['image']
    # Get all form data as a dictionary
    form_data = request.form.to_dict()

    # Access fields using .get() with defaults for robustness
    project_id = form_data.get('projectId')
    state_id = form_data.get('stateId')
    model_id = form_data.get('modelId')
    tool_name = form_data.get('toolName')

    if not all([project_id, state_id, model_id, tool_name]):
        print(f"DEBUG: Missing form data for inference: projectId={project_id}, stateId={state_id}, modelId={model_id}, toolName={tool_name}")
        # Provide more detail for debugging
        return jsonify({"success": False, "message": f"Incomplete project configuration details received by Flask backend. Missing one or more of: projectId, State ID, Model ID, Tool Name. Received form data: {form_data}"}), 400

    img_bytes = image_file.read()

    # 1. Register Job
    job_config = {
        'username': 'flask_user',
        'project_id': project_id,
        'state_id': state_id
    }
    register_response = external_ai_client.RegisterJobs(job_config)
    
    # FIX: Correctly check for 'job_id' in the nested 'data' and retrieve it
    if not register_response['success'] or 'job_id' not in register_response.get('data', {}).get('data', {}):
        # Fallback for more informative message if 'data' or inner 'data' is missing
        error_message = register_response.get('message', 'Unknown error during job registration.')
        if register_response.get('data') and not register_response.get('data').get('data'):
             error_message = "Job registration response missing nested 'data' for job_id."
        elif 'job_id' not in register_response.get('data', {}).get('data', {}):
             error_message = "Job ID not found in the job registration response."
             
        return jsonify({"success": False, "message": f"Failed to register job: {error_message}"}), 500
    
    job_id = register_response['data']['data']['job_id'] # FIX: Access nested 'data' to get job_id

    # 2. Run Inference
    inference_response = external_ai_client.inference(
        img_bytes=img_bytes,
        tool_name=tool_name,
        model_id=model_id,
        job_id=job_id,
        state_id=state_id
    )

    if not inference_response['success'] or not inference_response.get('data'):
        return jsonify({"success": False, "message": f"Inference failed: {inference_response.get('message', 'Unknown error')}"}), 500

    # Capture the full raw response from the AI server
    result_data_raw = inference_response['data'] 
    # Extract the actual inference results from the nested 'data' key
    actual_inference_data = result_data_raw.get('data', {}) 

    # --- NEW DEBUG PRINT: Print result_data content ---
    print(f"DEBUG: run_inference: result_data_raw received from AI server: {json.dumps(result_data_raw, indent=2)}")
    print(f"DEBUG: run_inference: actual_inference_data extracted: {json.dumps(actual_inference_data, indent=2)}")
    # --- END NEW DEBUG PRINT ---

    # Check if the AI server itself returned a processed base64 image (from result_data_raw)
    if 'processed_image_base64' in result_data_raw and result_data_raw['processed_image_base64']:
        print("DEBUG: AI server returned pre-processed base64 image. Using it directly.")
        return jsonify({"success": True, "processed_image_base64": result_data_raw['processed_image_base64'], "raw_data": result_data_raw})
    else:
        # Fallback to local drawing if server doesn't return base64 image
        print("DEBUG: AI server did not return pre-processed base64 image. Falling back to local drawing.")
        try:
            original_img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            # CRUCIAL: Convert PIL RGB image to OpenCV BGR format for drawing functions
            original_img_np_bgr = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR) 

            # Correctly access 'rois' from 'actual_inference_data'
            rois = np.array(actual_inference_data.get('rois', []))
            
            # --- NEW DEBUG PRINT: Print rois array content after numpy conversion ---
            print(f"DEBUG: run_inference: rois (NumPy array) before passing to drawing function: {rois}")
            # --- END NEW DEBUG PRINT ---

            # --- START: DIRECTLY ADAPTED MASK RENDERING LOGIC FROM YOUR EXAMPLECODE.PY ---
            # Pass 'actual_inference_data' to render_masks_from_server_response
            masks_decoded = render_masks_from_server_response(actual_inference_data, original_img_np_bgr.shape[:2])
            # If render_masks_from_server_response returns None, it means a severe error,
            # so we'll ensure masks_decoded is an empty list for safety.
            if masks_decoded is None: 
                print("ERROR: Mask decoding returned None, which indicates a critical issue during mask parsing. Providing dummy masks.")
                masks_decoded = [np.zeros(original_img_np_bgr.shape[:2], dtype=np.uint8)] * len(rois)
            # --- END: DIRECTLY ADAPTED MASK RENDERING LOGIC ---

            # Correctly access 'keypoints', 'class_ids', 'scores' from 'actual_inference_data'
            keypoints = np.array(actual_inference_data.get('keypoints', []))
            class_ids = np.array(actual_inference_data.get('class_ids', []))
            scores = np.array(actual_inference_data.get('scores', []))

            # Fetch project config again to get class names (if not cached)
            current_project_details = external_ai_client.get_single_project(project_id)
            if current_project_details and current_project_details['success'] and current_project_details['data'] and current_project_details['data'].get('data'):
                current_project_config = current_project_details['data'].get('data')[0]
            else:
                current_project_config = {}

            # CRITICAL FIX: getClassName now correctly extracts nested class_name
            class_name_list = getClassName(current_project_config) 

            # Max class ID determines required colors
            max_class_id_from_server = -1
            if len(class_ids) > 0:
                # Ensure class_ids are integers for max()
                cleaned_class_ids = [int(cid) for cid in class_ids if isinstance(cid, (int, float))]
                if cleaned_class_ids:
                    max_class_id_from_server = max(cleaned_class_ids)
            
            num_colors_to_generate = max(len(class_name_list), max_class_id_from_server + 1, 10)
            drawing_colors = random_colors(num_colors_to_generate)

            processed_image_base64 = display_instances_cv2_for_web(
                original_img_np_bgr, # This is already BGR
                rois, masks_decoded, keypoints, class_ids, scores, # Use the correctly extracted data
                class_name=class_name_list, colors=drawing_colors,
                show_mask=True # Ensure mask drawing is enabled if data is present
            )
            
            return jsonify({"success": True, "processed_image_base64": processed_image_base64, "raw_data": result_data_raw})

        except Exception as e:
            print(f"ERROR: Local drawing fallback failed: {e}")
            return jsonify({"success": False, "message": f"Inference completed, but local drawing failed: {e}. Raw data: {result_data_raw}"}), 500

# --- Frontend HTML/JS (Embedded) ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local AI Inference Center (Real-time)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .spinner {
            border: 4px solid rgba(0, 0, 0, 0.1);
            border-left-color: #ffffff;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        #videoElement, #outputCanvas {
            max-width: 100%;
            height: auto; /* Maintain aspect ratio */
            border-radius: 0.5rem; /* rounded-lg */
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-md */
            background-color: #000; /* Black background for video/canvas */
        }
    </style>
</head>
<body class="min-h-screen bg-gray-100 flex flex-col items-center p-4 text-gray-800">
    <div class="w-full max-w-4xl bg-white p-8 rounded-lg shadow-xl mb-8">
        <h1 class="text-3xl font-bold text-center text-indigo-700 mb-6">Local AI Inference Center (Real-time)</h1>

        <!-- Server Configuration -->
        <div class="mb-8 p-6 bg-indigo-50 rounded-lg shadow-inner">
            <h2 class="text-xl font-semibold text-indigo-600 mb-4">External AI Server Settings</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                <div>
                    <label for="serverIp" class="block text-sm font-medium text-gray-700 mb-1">Server IP Address</label>
                    <input type="text" id="serverIp" value="172.20.10.10" class="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" />
                </div>
                <div>
                    <label for="serverPort" class="block text-sm font-medium text-gray-700 mb-1">Server Port</label>
                    <input type="text" id="serverPort" value="5000" class="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" />
                </div>
                <div class="flex items-end">
                    <label class="inline-flex items-center">
                        <input type="checkbox" id="useHttps" class="form-checkbox h-5 w-5 text-indigo-600" checked />
                        <span class="ml-2 text-gray-700">Use HTTPS</span>
                    </label>
                </div>
            </div>
            <button id="connectButton" class="w-full py-2 px-4 rounded-md font-semibold text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2">
                Connect to AI Server & Fetch Models
            </button>
        </div>

        <!-- Status Messages -->
        <div id="loadingMessage" class="hidden p-3 mb-4 text-center bg-blue-100 rounded-md text-blue-800 flex items-center justify-center">
            <div class="spinner mr-3"></div>
            <span>Loading...</span>
        </div>
        <div id="errorMessage" class="hidden p-3 mb-4 bg-red-100 rounded-md text-red-800 text-center"></div>
        <div id="successMessage" class="hidden p-3 mb-4 bg-green-100 rounded-md text-green-800 text-center"></div>

        <!-- Model Selection -->
        <div class="mb-8 p-6 bg-gray-50 rounded-lg shadow-inner">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">Choose Model</h2>
            <select id="modelSelect" class="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" disabled>
                <option value="">-- Select a Model --</option>
            </select>
            <div id="modelInfo" class="mt-4 text-sm text-gray-600"></div>
        </div>

        <!-- Camera Selection -->
        <div class="mb-8 p-6 bg-gray-50 rounded-lg shadow-inner">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">Camera Selection</h2>
            <div class="flex items-center space-x-4 mb-4">
                <select id="cameraSelect" class="w-full p-2 border border-gray-300 rounded-md focus:ring-indigo-500 focus:border-indigo-500" disabled>
                    <option value="">-- Select Camera --</option>
                </select>
                <button id="refreshCamerasButton" class="py-2 px-4 rounded-md font-semibold text-white bg-gray-600 hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2">
                    Refresh Cameras
                </button>
            </div>
        </div>

        <!-- Camera and Detection Controls -->
        <div class="mb-8 p-6 bg-gray-50 rounded-lg shadow-inner text-center">
            <h2 class="text-xl font-semibold text-gray-700 mb-4">Real-time Detection</h2>
            <div class="flex justify-center items-center space-x-4 mb-4">
                <button id="startDetectionButton" class="py-3 px-6 rounded-md font-semibold text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2" disabled>
                    Start Camera & Detection
                </button>
                <button id="stopDetectionButton" class="py-3 px-6 rounded-md font-semibold text-gray-700 bg-gray-300 hover:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2" disabled>
                    Stop Detection
                </button>
            </div>
            <div class="flex flex-col md:flex-row justify-center items-center space-y-4 md:space-y-0 md:space-x-4 mt-4">
                <video id="videoElement" autoplay playsinline class="w-full md:w-1/2"></video>
                <canvas id="outputCanvas" class="w-full md:w-1/2"></canvas>
                <canvas id="captureCanvas" style="display:none;"></canvas> <!-- Hidden canvas for capturing frames -->
            </div>
        </div>

        <!-- Inference Results Display -->
        <div id="inferenceResultContainer" class="mb-8 p-6 bg-green-50 rounded-lg shadow-inner">
            <h2 class="text-xl font-semibold text-green-700 mb-4 text-center">Inference Result Data</h2>
            <div class="mt-4 p-3 bg-green-100 rounded-md text-sm text-green-800">
                <h3 class="font-medium mb-2">Objects Detected:</h3>
                <p id="objectsDetectedCount">0 object(s) detected.</p>
            </div>
             <div class="mt-4 p-3 bg-green-100 rounded-md text-sm text-green-800 overflow-auto max-h-60">
                <h3 class="font-medium mb-2">Raw Inference Data (Last Frame):</h3>
                <pre id="inferenceResultData" class="whitespace-pre-wrap break-all text-xs"></pre>
            </div>
        </div>

        <div class="mt-8 text-center text-gray-500 text-sm">
            
        </div>
    </div>

    <script>
        // --- UI Element References ---
        const serverIpInput = document.getElementById('serverIp');
        const serverPortInput = document.getElementById('serverPort');
        const useHttpsCheckbox = document.getElementById('useHttps');
        const connectButton = document.getElementById('connectButton');
        const modelSelect = document.getElementById('modelSelect');
        const modelInfoDiv = document.getElementById('modelInfo');
        
        const cameraSelect = document.getElementById('cameraSelect'); // New
        const refreshCamerasButton = document.getElementById('refreshCamerasButton'); // New

        const startDetectionButton = document.getElementById('startDetectionButton');
        const stopDetectionButton = document.getElementById('stopDetectionButton');
        const videoElement = document.getElementById('videoElement');
        const captureCanvas = document.getElementById('captureCanvas');
        const outputCanvas = document.getElementById('outputCanvas');
        const outputCtx = outputCanvas.getContext('2d'); // Get 2D context for output canvas

        const inferenceResultContainer = document.getElementById('inferenceResultContainer');
        const objectsDetectedCount = document.getElementById('objectsDetectedCount');
        const inferenceResultDataPre = document.getElementById('inferenceResultData');

        const loadingMessage = document.getElementById('loadingMessage');
        const errorMessage = document.getElementById('errorMessage');
        const successMessage = document.getElementById('successMessage');

        // --- Global State Variables ---
        let allProjectsData = [];
        let currentSelectedProjectConfig = null;
        let videoStream = null; // To hold the MediaStream from camera
        let detectionInterval = null; // To hold the setInterval ID for real-time detection loop
        let isFlaskClientInitialized = false; // Flag to track if Flask's client is initialized
        const FRAME_RATE_MS = 100; // Adjusted for smoother real-time feel (10 frames per second)
        let isProcessingFrame = false; // New flag to prevent overlapping inference requests
        let selectedDeviceId = ''; // New: Stores the deviceId of the selected camera

        // --- UI Feedback Functions ---
        function showLoading(msg = 'Loading...') {
            loadingMessage.classList.remove('hidden');
            loadingMessage.querySelector('span').textContent = msg;
            hideError();
            hideSuccess();
            updateButtonStates();
        }

        function hideLoading() {
            loadingMessage.classList.add('hidden');
            updateButtonStates();
        }

        function showError(msg) {
            errorMessage.classList.remove('hidden');
            errorMessage.textContent = `Error: ${msg}`;
            hideLoading();
            hideSuccess();
            updateButtonStates();
        }

        function hideError() {
            errorMessage.classList.add('hidden');
        }

        function showSuccess(msg) {
            successMessage.classList.remove('hidden');
            successMessage.textContent = msg;
            hideLoading();
            hideError();
            updateButtonStates();
        }

        function hideSuccess() {
            successMessage.classList.add('hidden');
        }

        function updateButtonStates() {
            const isLoading = !loadingMessage.classList.contains('hidden');
            const isConnected = isFlaskClientInitialized; 
            const modelSelected = !!currentSelectedProjectConfig;
            const isCameraSelected = !!selectedDeviceId; // New: Check if a camera is selected
            const isCameraRunning = !!videoStream;
            const isDetectionRunning = !!detectionInterval;

            modelSelect.disabled = isLoading || !isConnected || allProjectsData.length === 0;
            cameraSelect.disabled = isLoading || !isConnected; // Enable camera select only if connected
            
            // Enable Start button only if connected, model selected, camera selected, and camera/detection not running
            startDetectionButton.disabled = isLoading || !isConnected || !modelSelected || !isCameraSelected || isCameraRunning || isDetectionRunning;
            startDetectionButton.classList.toggle('bg-indigo-600', !startDetectionButton.disabled);
            startDetectionButton.classList.toggle('hover:bg-indigo-700', !startDetectionButton.disabled);
            startDetectionButton.classList.toggle('focus:outline-none', !startDetectionButton.disabled);
            startDetectionButton.classList.toggle('focus:ring-2', !startDetectionButton.disabled);
            startDetectionButton.classList.toggle('focus:ring-indigo-500', !startDetectionButton.disabled);
            startDetectionButton.classList.toggle('focus:ring-offset-2', !startDetectionButton.disabled);
            startDetectionButton.classList.toggle('bg-indigo-300', startDetectionButton.disabled);
            startDetectionButton.classList.toggle('cursor-not-allowed', startDetectionButton.disabled);

            // Enable Stop button only if camera/detection is running
            stopDetectionButton.disabled = isLoading || !isCameraRunning || !isDetectionRunning;
            stopDetectionButton.classList.toggle('bg-red-600', !stopDetectionButton.disabled);
            stopDetectionButton.classList.toggle('hover:bg-red-700', !stopDetectionButton.disabled);
            stopDetectionButton.classList.toggle('focus:outline-none', !stopDetectionButton.disabled);
            stopDetectionButton.classList.toggle('focus:ring-2', !stopDetectionButton.disabled);
            stopDetectionButton.classList.toggle('focus:ring-red-500', !stopDetectionButton.disabled);
            stopDetectionButton.classList.toggle('focus:ring-offset-2', !stopDetectionButton.disabled);
            stopDetectionButton.classList.toggle('bg-gray-300', stopDetectionButton.disabled);
            stopDetectionButton.classList.toggle('cursor-not-allowed', stopDetectionButton.disabled);
        }

        function clearInferenceResultsDisplay() {
            outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
            objectsDetectedCount.textContent = '0 object(s) detected.';
            inferenceResultDataPre.textContent = '';
        }

        // --- Camera and Detection Logic ---

        async function populateCameraSelect() {
            cameraSelect.innerHTML = '<option value="">-- Select Camera --</option>';
            try {
                // Request camera access first to get labels for devices
                // This is a common trick: browsers hide device labels until permission is granted
                const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                const devices = await navigator.mediaDevices.enumerateDevices();
                
                tempStream.getTracks().forEach(track => track.stop()); // Stop the temporary stream

                let hasCameras = false;
                devices.forEach(device => {
                    if (device.kind === 'videoinput') {
                        const option = document.createElement('option');
                        option.value = device.deviceId;
                        option.textContent = device.label || `Camera ${cameraSelect.options.length}`;
                        cameraSelect.appendChild(option);
                        hasCameras = true;
                    }
                });

                if (!hasCameras) {
                    cameraSelect.innerHTML = '<option value="">-- No Cameras Found --</option>';
                } else {
                    // Try to pre-select the previously selected device, or the first one
                    if (selectedDeviceId && Array.from(cameraSelect.options).some(opt => opt.value === selectedDeviceId)) {
                        cameraSelect.value = selectedDeviceId;
                    } else if (cameraSelect.options.length > 1) { // If more than just the "-- Select Camera --" option
                        cameraSelect.value = cameraSelect.options[1].value; // Select the first actual camera
                        selectedDeviceId = cameraSelect.options[1].value;
                    }
                }
                updateButtonStates();
            } catch (err) {
                console.error("Error enumerating devices or accessing camera for labels:", err);
                showError(`Failed to list cameras: ${err.message}. Please ensure camera permissions are granted.`);
                cameraSelect.innerHTML = '<option value="">-- Error Listing Cameras --</option>';
                cameraSelect.disabled = true;
                updateButtonStates();
            }
        }

        async function startCamera() {
            if (!selectedDeviceId) {
                showError("Please select a camera from the dropdown first.");
                return;
            }

            showLoading('Starting camera...');
            clearInferenceResultsDisplay();
            try {
                // Use the selected deviceId in getUserMedia constraints
                videoStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        deviceId: { exact: selectedDeviceId }
                    }
                });
                videoElement.srcObject = videoStream;
                await videoElement.play();

                // Set canvas dimensions to match video dimensions
                outputCanvas.width = videoElement.videoWidth;
                outputCanvas.height = videoElement.videoHeight;
                captureCanvas.width = videoElement.videoWidth;
                captureCanvas.height = videoElement.videoHeight;

                showSuccess('Camera started. Starting real-time detection...');
                startDetectionLoop();

            } catch (err) {
                console.error("Error accessing camera:", err);
                showError(`Failed to start camera: ${err.message}. Make sure you've granted camera permissions and selected a valid camera.`);
                stopCamera();
            } finally {
                hideLoading();
            }
        }

        function stopCamera() {
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop()); // Stop all tracks
                videoElement.srcObject = null;
                videoStream = null;
            }
            stopDetectionLoop();
            clearInferenceResultsDisplay();
            showSuccess('Camera and detection stopped.');
        }

        function startDetectionLoop() {
            if (detectionInterval) clearInterval(detectionInterval); // Clear any existing interval
            
            // Draw initial frame to canvas to ensure it's not blank
            const captureCtx = captureCanvas.getContext('2d');
            captureCtx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);

            detectionInterval = setInterval(async () => {
                if (isProcessingFrame) { // If a previous frame is still being processed, skip this one
                    // console.log("Skipping frame: Previous frame still processing."); // Can be noisy, uncomment for deep debugging
                    return;
                }

                if (!videoElement.paused && !videoElement.ended && videoElement.videoWidth > 0) {
                    isProcessingFrame = true; // Set flag to true to indicate processing has started

                    captureCtx.drawImage(videoElement, 0, 0, captureCanvas.width, captureCanvas.height);

                    // Convert canvas to Blob (more efficient than base64 for upload)
                    captureCanvas.toBlob(async (blob) => {
                        if (!blob) {
                            console.error("Failed to get blob from canvas.");
                            isProcessingFrame = false; // Reset flag
                            return;
                        }

                        if (!currentSelectedProjectConfig) {
                            console.warn("No model selected. Skipping inference for this frame.");
                            isProcessingFrame = false; // Reset flag
                            return;
                        }

                        const formData = new FormData();
                        formData.append('image', blob, 'camera_frame.jpeg'); // Pass as a file
                        formData.append('projectId', currentSelectedProjectConfig.project_id || '');
                        formData.append('stateId', currentSelectedProjectConfig.state_id || '');
                        formData.append('modelId', currentSelectedProjectConfig.model?.model_id || ''); 
                        formData.append('toolName', currentSelectedProjectConfig.tool || '');

                        try {
                            const response = await fetch('/api/inference', {
                                method: 'POST',
                                body: formData
                            });
                            const result = await response.json();

                            if (result.success && result.processed_image_base64) {
                                const img = new Image();
                                img.onload = () => {
                                    outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height); // Clear previous drawings
                                    outputCtx.drawImage(img, 0, 0, outputCanvas.width, outputCanvas.height); // Draw processed image
                                };
                                img.src = `data:image/jpeg;base64,${result.processed_image_base64}`;
                                
                                const objectsCount = result.raw_data.data?.rois?.length || 0;
                                objectsDetectedCount.textContent = `${objectsCount} object(s) detected.`;
                                inferenceResultDataPre.textContent = JSON.stringify(result.raw_data, null, 2);

                            } else {
                                console.error("Inference failed for frame:", result.message || 'Unknown error.');
                                objectsDetectedCount.textContent = `Error: ${result.message || 'Unknown error.'}`;
                            }
                        } catch (err) {
                            console.error("Error sending frame for inference:", err);
                            objectsDetectedCount.textContent = `Error: ${err.message}`;
                        } finally {
                            isProcessingFrame = false; // Always reset flag after processing is complete (or error)
                        }
                    }, 'image/jpeg', 0.9); // Image format and quality (0.9 for better quality, reduces compression artifacts)
                }
            }, FRAME_RATE_MS); // Send frame every FRAME_RATE_MS
        }

        function stopDetectionLoop() {
            if (detectionInterval) {
                clearInterval(detectionInterval);
                detectionInterval = null;
            }
        }


        // --- API Calls to Flask Backend (mostly unchanged) ---

        async function initializeClientAndFetchProjects() {
            showLoading('Connecting to AI server and fetching models...');
            stopCamera(); // Ensure camera is off before reconnecting
            try {
                const initResponse = await fetch('/api/init', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        serverIp: serverIpInput.value,
                        serverPort: serverPortInput.value,
                        useHttps: useHttpsCheckbox.checked
                    })
                });
                const initResult = await initResponse.json();

                if (!initResult.success) {
                    isFlaskClientInitialized = false; // Set flag to false on failure
                    throw new Error(`Failed to initialize client: ${initResult.message}`);
                }
                isFlaskClientInitialized = true; // Set flag to true on success

                const projectsResponse = await fetch('/api/projects');
                const projectsResult = await projectsResponse.json();

                if (!projectsResult.success) {
                    isFlaskClientInitialized = false; // Revert flag if project fetch fails
                    throw new Error(`Failed to fetch projects: ${projectsResult.message}`);
                }
                
                allProjectsData = projectsResult.data.data; 
                populateModelSelect();
                showSuccess('Successfully connected and loaded models!');

            } catch (err) {
                isFlaskClientInitialized = false; // Ensure false on any error in this flow
                showError(`Connection/Fetch Error: ${err.message}. Please check server settings, server logs, and console for more details.`);
                modelSelect.disabled = true;
                modelSelect.innerHTML = '<option value="">-- No Models Found --</option>';
            } finally {
                hideLoading();
                populateCameraSelect(); // Try to populate cameras once connected
            }
        }

        async function fetchSelectedModelConfig(projectId) {
            showLoading('Fetching model configuration...');
            clearInferenceResultsDisplay(); // Clear results on new model select
            modelInfoDiv.textContent = '';

            try {
                const response = await fetch(`/api/project/${projectId}`);
                const result = await response.json();

                if (result.success && Array.isArray(result.data) && result.data.length > 0) { 
                    currentSelectedProjectConfig = result.data[0]; 
                    currentSelectedProjectConfig.project_id = projectId; 
                    
                    const modelName = currentSelectedProjectConfig.model?.name || 'N/A';
                    const toolName = currentSelectedProjectConfig.tool || 'N/A';
                    const showMaskConfig = currentSelectedProjectConfig.model?.config?.[0]?.show_mask ? 'True' : 'False';

                    modelInfoDiv.innerHTML = `
                        <strong>Tool:</strong> ${toolName}<br/>
                        <strong>Model Name:</strong> ${modelName}<br/>
                        <strong>Server Config: Show Mask:</strong> ${showMaskConfig}
                    `;
                    showSuccess(`Configuration loaded for ${modelName}.`);
                } else {
                    currentSelectedProjectConfig = null;
                    const msg = result.message || (result.data && result.data.message) || 'No data received or data format incorrect.';
                    showError(`Failed to load config for project ID ${projectId}: ${msg}`);
                    console.error("DEBUG JS: Full response for single project config failure:", result);
                }
            } catch (err) {
                currentSelectedProjectConfig = null;
                showError(`Error fetching model configuration: ${err.message}`);
            } finally {
                hideLoading();
            }
        }

        // --- UI Population Functions ---
        function populateModelSelect() {
            modelSelect.innerHTML = '<option value="">-- Select a Model --</option>';
            if (Array.isArray(allProjectsData) && allProjectsData.length > 0) {
                allProjectsData.forEach(p => {
                    if (p && typeof p.project_id !== 'undefined' && typeof p.project_name !== 'undefined') {
                        const option = document.createElement('option');
                        option.value = p.project_id;
                        option.textContent = p.project_name;
                        modelSelect.appendChild(option);
                    }
                });
                modelSelect.disabled = false;
            } else {
                modelSelect.disabled = true;
                modelSelect.innerHTML = '<option value="">-- No Models Available --</option>';
            }
            if (Array.isArray(allProjectsData) && allProjectsData.length === 1 && allProjectsData[0] && typeof allProjectsData[0].project_id !== 'undefined') {
                modelSelect.value = allProjectsData[0].project_id; 
                fetchSelectedModelConfig(allProjectsData[0].project_id);
            }
            updateButtonStates();
        }

        // --- Event Listeners ---
        connectButton.addEventListener('click', initializeClientAndFetchProjects);

        modelSelect.addEventListener('change', (event) => {
            const projectId = event.target.value;
            if (projectId) {
                fetchSelectedModelConfig(projectId);
            } else {
                currentSelectedProjectConfig = null;
                modelInfoDiv.textContent = '';
                stopCamera(); // Stop camera if model deselected
            }
            updateButtonStates();
        });

        cameraSelect.addEventListener('change', (event) => { // New: Update selectedDeviceId
            selectedDeviceId = event.target.value;
            stopCamera(); // Stop camera if camera selection changes
            updateButtonStates();
        });

        refreshCamerasButton.addEventListener('click', populateCameraSelect); // New: Refresh cameras

        startDetectionButton.addEventListener('click', startCamera);
        stopDetectionButton.addEventListener('click', stopCamera);

        // --- Initial Load ---
        document.addEventListener('DOMContentLoaded', () => {
            updateButtonStates(); // Set initial button states
            // We'll call populateCameraSelect after successful connection to AI server
            // or you can uncomment below to allow listing cameras before connection
            // populateCameraSelect(); 
        });
    </script>
</body>
</html>
""" # END OF HTML_TEMPLATE

# --- Main App Execution ---
if __name__ == '__main__':
    print("Starting Flask Local AI Inference Center...")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("Then, enter your AI server details and click 'Connect'.")
    # Run the Flask app on localhost, port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)
