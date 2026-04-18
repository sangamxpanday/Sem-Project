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
        if os.path.exists("models/vehicle_classifier_final.pt"):
            CLASSIFIER = YOLO("models/vehicle_classifier_final.pt")
            print("✅ Classifier model loaded")
        else:
            print("⚠️  Classifier model not found at models/vehicle_classifier_final.pt")
    except Exception as e:
        print(f"❌ Failed to load classifier: {e}")

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]
TRACKER_CONFIG = "bytetrack.yaml"


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
    if not DETECTOR:
        print("❌ Detector not available")
        return None
    
    results_data = []
    
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Processing video: {frame_count} frames at {fps} FPS")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate video timestamp
            video_timestamp_sec = frame_idx / fps if fps > 0 else 0
            
            # Calculate actual detection time
            detection_time = upload_time + timedelta(seconds=video_timestamp_sec)
            
            # Run detection + tracking (persist IDs across frames)
            try:
                results = DETECTOR.track(
                    frame,
                    persist=True,
                    verbose=False,
                    conf=0.5,
                    tracker=TRACKER_CONFIG
                )
            except Exception:
                # Fallback to plain detection if tracker has an environment issue
                results = DETECTOR(frame, verbose=False, conf=0.5)
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = DETECTOR.names.get(cls_id, "unknown")
                    
                    # Only process vehicle classes
                    if label in VEHICLE_CLASSES:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        track_id = int(box.id[0]) if box.id is not None else None
                        
                        # Try to classify if classifier available
                        vehicle_type = label
                        if CLASSIFIER:
                            try:
                                h, w = frame.shape[:2]
                                x1_clip, y1_clip = max(0, x1), max(0, y1)
                                x2_clip, y2_clip = min(w, x2), min(h, y2)
                                crop = frame[y1_clip:y2_clip, x1_clip:x2_clip]
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
                            'track_id': track_id,
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'frame': frame_idx,
                            'video_source': video_filename
                        })
            
            frame_idx += 1
            if frame_idx % max(1, frame_count // 10) == 0:
                print(f"  Progress: {frame_idx}/{frame_count} frames processed")
        
        cap.release()
        
        print(f"✅ Detected {len(results_data)} vehicles")
        
        # Create DataFrame
        if results_data:
            df = pd.DataFrame(results_data)
        else:
            df = pd.DataFrame(columns=[
                'date', 'time', 'video_timestamp', 'vehicle_type', 'confidence',
                'track_id', 'x1', 'y1', 'x2', 'y2', 'frame', 'video_source'
            ])
        
        # Generate CSV filename with date and video name (clean filename)
        video_name_clean = Path(video_filename).stem  # Remove extension
        csv_filename = f"{upload_time.strftime('%Y%m%d_%H%M%S')}_{video_name_clean}_traffic.csv"
        csv_path = os.path.join(app.config['RESULTS_FOLDER'], csv_filename)
        
        # Save CSV
        df.to_csv(csv_path, index=False)
        print(f"💾 Results saved to: {csv_path}")
        
        # Also save metadata mapping
        metadata = {
            'csv_file': csv_filename,
            'video_file': video_filename,
            'capture_time': upload_time.isoformat(),
            'detections': len(results_data)
        }
        save_result_metadata(metadata)
        
        return csv_path
    
    except Exception as e:
        print(f"❌ Error processing video: {e}")
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
    """Run video processing in background thread."""
    thread = threading.Thread(target=process_video_inference, args=(video_path, upload_time, video_filename))
    thread.daemon = True
    thread.start()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET'])
def upload_page():
    videos = sorted(os.listdir(app.config['UPLOAD_FOLDER']))
    return render_template('upload.html', videos=videos)


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('❌ No file part')
        return redirect(url_for('upload_page'))

    file = request.files['video']
    if file.filename == '':
        flash('❌ No selected file')
        return redirect(url_for('upload_page'))

    # Get date and time from form
    upload_date_str = request.form.get('upload_date')
    upload_time_str = request.form.get('upload_time')
    
    if not upload_date_str or not upload_time_str:
        flash('❌ Please provide both date and time')
        return redirect(url_for('upload_page'))
    
    try:
        # Parse date and time
        upload_time = datetime.strptime(f"{upload_date_str} {upload_time_str}", "%Y-%m-%d %H:%M")
    except ValueError as e:
        flash('❌ Invalid date/time format')
        return redirect(url_for('upload_page'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(dest_path)
        
        # Use form date/time
        flash(f'✅ Uploaded: {filename} at {upload_time.strftime("%Y-%m-%d %H:%M:%S")}')
        
        # Start background processing
        if YOLO_AVAILABLE and DETECTOR:
            print(f"🎬 Starting model inference on: {filename}")
            print(f"   Capture time: {upload_time.strftime('%Y-%m-%d %H:%M:%S')}")
            process_video_async(dest_path, upload_time, filename)
            flash(f'📊 Model inference started (results will be saved to CSV)')
        else:
            flash(f'⚠️  Model inference not available (YOLO not loaded)')
        
        return redirect(url_for('upload_page'))

    flash('❌ Invalid video format. Allowed: ' + ', '.join(sorted(ALLOWED_EXTENSIONS)))
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
