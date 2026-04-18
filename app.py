from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from pathlib import Path
import os
import cv2
import pandas as pd
import threading
import json

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not installed. Install with: pip install ultralytics opencv-python")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads', 'videos')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
ALLOWED_EXTENSIONS = {'mp4', 'webm', 'mkv', 'avi', 'mov'}

app = Flask(__name__)
app.secret_key = 'change-this-secret'

# Configure file upload limits (1GB)
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Load models
DETECTOR = None
CLASSIFIER = None

if YOLO_AVAILABLE:
    try:
        print("📦 Loading detection model...")
        DETECTOR = YOLO("yolov8n.pt")
        print("✅ Detection model loaded")
    except Exception as e:
        print(f"❌ Failed to load detector: {e}")
    
    try:
        print("📦 Loading classifier model...")
        print(f"   Current working directory: {os.getcwd()}")
        
        classifier_paths = [
            "models/vehicle_classifier_final.pt",
            "/app/models/vehicle_classifier_final.pt",
            "./models/vehicle_classifier_final.pt",
        ]
        
        classifier_found = False
        for path in classifier_paths:
            if os.path.exists(path):
                size = os.path.getsize(path)
                print(f"   ✓ Found custom classifier at {path} ({size} bytes)")
                try:
                    CLASSIFIER = YOLO(path)
                    print(f"✅ Classifier model loaded from {path}")
                    classifier_found = True
                    break
                except Exception as load_err:
                    err_msg = str(load_err)[:150]
                    if "ultralytics.nn.modules" in err_msg:
                        print(f"   ⚠️  Custom model incompatible (version mismatch): {err_msg}")
                    else:
                        print(f"   ✗ Error loading {path}: {err_msg}")
        
        if not classifier_found:
            print(f"⚠️  Using generic YOLOv8n-cls classifier (custom model not available)")
            try:
                CLASSIFIER = YOLO("yolov8n-cls.pt")
                print(f"✅ Loaded generic YOLOv8n-cls as fallback")
            except Exception as e:
                print(f"⚠️  Classifier not available: {e}")
            
    except Exception as e:
        print(f"❌ Failed to load classifier: {e}")

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]


# Error handlers
@app.errorhandler(400)
def bad_request(error):
    print(f"❌ 400 Bad Request: {error}")
    flash('❌ Invalid form data. Please try again.')
    return redirect(url_for('upload_page')), 400


@app.errorhandler(413)
def request_entity_too_large(error):
    print(f"❌ 413 File too large: {error}")
    flash('❌ File too large. Maximum size is 1GB.')
    return redirect(url_for('upload_page')), 413


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video_inference(video_path, upload_time, video_filename):
    """
    Process video: run inference, calculate timestamps, create CSV.
    video_path: full path to video file
    upload_time: datetime object when user captured video
    video_filename: original video filename for tracking
    Returns: path to results CSV
    """
    print(f"   [1] Checking DETECTOR...")
    if not DETECTOR:
        print(f"   ❌ [1] Detector not available - STOPPING")
        return None
    
    print(f"   ✅ [1] Detector available")
    results_data = []
    
    try:
        # Open video
        print(f"   [2] Opening video file...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   ✅ [2] Video opened: {frame_count} frames at {fps} FPS")
        print(f"       File: {video_path}")
        print(f"       Size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
        
        if frame_count == 0 or fps == 0:
            print(f"   ❌ [2] Invalid video: {frame_count} frames, {fps} FPS - STOPPING")
            return None
        
        print(f"   [3] Starting frame processing...")
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate video timestamp
            video_timestamp_sec = frame_idx / fps if fps > 0 else 0
            
            # Calculate actual detection time
            detection_time = upload_time + timedelta(seconds=video_timestamp_sec)
            
            # Run detection
            results = DETECTOR(frame, verbose=False, conf=0.3)
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = DETECTOR.names.get(cls_id, "unknown")
                    
                    # Only process vehicle classes
                    if label in VEHICLE_CLASSES:
                        conf = float(box.conf[0])
                        
                        # Try to classify if classifier available
                        vehicle_type = label
                        if CLASSIFIER:
                            try:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                crop = frame[y1:y2, x1:x2]
                                if crop.size > 0:
                                    class_results = CLASSIFIER(crop, verbose=False)
                                    if class_results[0].probs:
                                        top_class_id = class_results[0].probs.top1
                                        top_class_name = class_results[0].names[top_class_id]
                                        vehicle_type = f"{label}_{top_class_name}"
                            except:
                                pass  # Use default label if classification fails
                        
                        results_data.append({
                            'date': detection_time.strftime('%Y-%m-%d'),
                            'time': detection_time.strftime('%H:%M:%S'),
                            'video_timestamp': f"{int(video_timestamp_sec//60)}:{int(video_timestamp_sec%60):02d}",
                            'vehicle_type': vehicle_type,
                            'confidence': round(conf, 4),
                            'frame': frame_idx,
                            'video_source': video_filename
                        })
            
            frame_idx += 1
            if frame_idx % max(1, frame_count // 10) == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"       ⏳ Progress: {frame_idx}/{frame_count} frames ({progress:.0f}%)")
        
        cap.release()
        
        print(f"   ✅ [3] Frame processing complete")
        print(f"   📊 [4] Detected {len(results_data)} vehicles total")
        
        # Create DataFrame
        print(f"   [5] Creating CSV...")
        df = pd.DataFrame(results_data)
        
        # Generate CSV filename with date and video name (clean filename)
        video_name_clean = Path(video_filename).stem  # Remove extension
        csv_filename = f"{upload_time.strftime('%Y%m%d_%H%M%S')}_{video_name_clean}_traffic.csv"
        csv_path = os.path.join(app.config['RESULTS_FOLDER'], csv_filename)
        
        print(f"       CSV path: {csv_path}")
        print(f"       Results folder exists: {os.path.exists(app.config['RESULTS_FOLDER'])}")
        
        # Save CSV
        print(f"   [6] Saving CSV file...")
        df.to_csv(csv_path, index=False)
        
        file_exists = os.path.exists(csv_path)
        file_size = os.path.getsize(csv_path) if file_exists else 0
        print(f"   ✅ [6] Results saved to: {csv_path}")
        print(f"       File exists: {file_exists}")
        print(f"       File size: {file_size} bytes")
        
        # Also save metadata mapping
        print(f"   [7] Saving metadata...")
        metadata = {
            'csv_file': csv_filename,
            'video_file': video_filename,
            'capture_time': upload_time.isoformat(),
            'detections': len(results_data)
        }
        save_result_metadata(metadata)
        print(f"   ✅ [7] Metadata saved")
        
        return csv_path
    
    except Exception as e:
        print(f"   ❌ Error in process_video_inference: {type(e).__name__}")
        print(f"      Message: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_result_metadata(metadata):
    """Save metadata mapping of video to CSV for tracking."""
    meta_file = os.path.join(app.config['RESULTS_FOLDER'], 'results_metadata.json')
    try:
        import json
        # Load existing metadata
        all_metadata = []
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                all_metadata = json.load(f)
        
        # Append new entry
        all_metadata.append(metadata)
        
        # Save back
        with open(meta_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        print(f"📝 Metadata saved")
    except Exception as e:
        print(f"⚠️  Could not save metadata: {e}")


def process_video_async(video_path, upload_time, video_filename):
    """Run video processing in background thread with error handling."""
    def thread_wrapper():
        try:
            print(f"\n{'='*60}")
            print(f"🔄 BACKGROUND PROCESSING STARTED for: {video_filename}")
            print(f"   Video path: {video_path}")
            print(f"   Capture time: {upload_time}")
            print(f"   File exists: {os.path.exists(video_path)}")
            if os.path.exists(video_path):
                print(f"   File size: {os.path.getsize(video_path) / (1024*1024):.1f} MB")
            print(f"{'='*60}\n")
            
            result = process_video_inference(video_path, upload_time, video_filename)
            
            if result:
                print(f"\n✅ BACKGROUND PROCESSING COMPLETED SUCCESSFULLY")
                print(f"   Results saved to: {result}")
                csv_exists = os.path.exists(result)
                print(f"   CSV file exists: {csv_exists}")
                if csv_exists:
                    print(f"   CSV size: {os.path.getsize(result)} bytes")
                print(f"\n")
            else:
                print(f"\n⚠️  BACKGROUND PROCESSING RETURNED NO RESULTS\n")
        except Exception as e:
            print(f"\n❌ BACKGROUND THREAD ERROR: {type(e).__name__}")
            print(f"   Message: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n")
    
    thread = threading.Thread(target=thread_wrapper)
    thread.daemon = True
    thread.start()
    print(f"📌 Background thread started (check logs for progress)")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET'])
def upload_page():
    videos = sorted(os.listdir(app.config['UPLOAD_FOLDER']))
    return render_template('upload.html', videos=videos)


@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            flash('❌ No video file selected')
            return redirect(url_for('upload_page'))

        file = request.files['video']
        if file.filename == '':
            flash('❌ No selected file')
            return redirect(url_for('upload_page'))

        upload_date_str = request.form.get('upload_date')
        upload_time_str = request.form.get('upload_time')
        
        if not upload_date_str or not upload_time_str:
            flash('❌ Please provide both date and time')
            return redirect(url_for('upload_page'))
        
        try:
            upload_time = datetime.strptime(f"{upload_date_str} {upload_time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            flash('❌ Invalid date/time format')
            return redirect(url_for('upload_page'))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(dest_path)
            
            print(f"✅ Video uploaded: {filename} ({os.path.getsize(dest_path)} bytes)")
            flash(f'✅ Uploaded: {filename}')
            
            if YOLO_AVAILABLE and DETECTOR:
                print(f"🎬 Starting inference: {filename}")
                process_video_async(dest_path, upload_time, filename)
                flash(f'📊 Processing started...')
            
            return redirect(url_for('upload_page'))

        flash('❌ Invalid video format. Allowed: mp4, webm, mkv, avi, mov')
        return redirect(url_for('upload_page'))
        
    except Exception as e:
        print(f"❌ Upload error: {type(e).__name__}: {str(e)[:200]}")
        flash(f'❌ Upload failed')
        return redirect(url_for('upload_page'))


@app.route('/uploads/videos/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results')
def results_list():
    """List all result CSV files with video source tracking."""
    csv_files = sorted(os.listdir(app.config['RESULTS_FOLDER']))
    
    # Filter out metadata file
    csv_files = [f for f in csv_files if f.endswith('.csv')]
    
    # Load metadata for tracking
    metadata_dict = {}
    meta_file = os.path.join(app.config['RESULTS_FOLDER'], 'results_metadata.json')
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r') as f:
                metadata_list = json.load(f)
                for item in metadata_list:
                    metadata_dict[item['csv_file']] = item
        except:
            pass
    
    # Prepare data with metadata
    results_data = []
    for csv in csv_files:
        meta = metadata_dict.get(csv, {})
        results_data.append({
            'csv_file': csv,
            'video_file': meta.get('video_file', 'Unknown'),
            'capture_time': meta.get('capture_time', 'N/A'),
            'detections': meta.get('detections', '?'),
            'file_size': os.path.getsize(os.path.join(app.config['RESULTS_FOLDER'], csv)) // 1024 if os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], csv)) else 0
        })
    
    return render_template('results.html', results=results_data)


@app.route('/results/<filename>')
def download_result(filename):
    """Download or view result CSV."""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


@app.route('/api/status')
def api_status():
    """API endpoint for system status."""
    return {
        'status': 'running',
        'yolo_available': YOLO_AVAILABLE,
        'detector_loaded': DETECTOR is not None,
        'classifier_loaded': CLASSIFIER is not None,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'results_folder': app.config['RESULTS_FOLDER']
    }


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚗 VEHICLE DETECTION NEPAL - Starting Application")
    print("="*60)
    print(f"🔧 YOLO Available: {YOLO_AVAILABLE}")
    print(f"🤖 Detector Loaded: {DETECTOR is not None}")
    print(f"📊 Classifier Loaded: {CLASSIFIER is not None}")
    print("="*60 + "\n")
    
    import os
    port = int(os.environ.get('PORT', 7860))
    debug = os.environ.get('DEBUG', 'False') == 'True'
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
