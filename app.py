import os
import cv2
import time
import base64
import asyncio
import aiohttp

import serial

import json
import threading
from flask import Flask, render_template, Response, jsonify, request, flash, redirect, url_for
from picamera2 import Picamera2
from ultralytics import YOLO
import subprocess

import csv
CSV_FILE = 'temp.csv'

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret')
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'RGB888', "size": (1280, 720)}))
picam2.start()

port = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)

model = YOLO("best.pt")
results_data = []

lock = threading.Lock()

async def send_to_ocr(session, image_base64, class_name, confidence, coor):
    url = "https://sniper-ocr-prod-wbebaohduc.ap-southeast-5.fcapp.run/ocr-plate"
    payload = json.dumps({"image_base64": image_base64})
    headers = {'Content-Type': 'application/json'}
    
    try:
        async with session.post(url, headers=headers, data=payload) as response:
            resp_json = await response.json()
            print(resp_json)
            print("Confidence: " + str(confidence))
            return {
                "confidence": round(confidence, 2),
                "image_base64": image_base64,
                "ocr": "success" if "success" in str(resp_json) else "fail",
                "coor": coor
            }
    except Exception:
        return {
            "confidence": round(confidence, 2),
            "image_base64": image_base64,
            "ocr": "fail",
            "coor": coor
        }

async def detect_async():
    global results_data
    async with aiohttp.ClientSession() as session:
        while True:
            frame = picam2.capture_array()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(image)
            boxes = results[0].boxes

            tasks = []

            if boxes:
                line = port.readline().decode('utf-8', errors='ignore')

                    
            for box in boxes:
                conf = float(box.conf[0])
                
                if conf < conf_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                cropped = frame[y1:y2, x1:x2]
                _, buffer = cv2.imencode('.jpg', cropped)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                tasks.append(send_to_ocr(session, jpg_as_text, class_name, conf, line))

            local_results = await asyncio.gather(*tasks)

            with lock:
                results_data = local_results


def detect_loop():
    asyncio.run(detect_async())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/control')
def control():
    return render_template('control.html', conf_threshold=conf_threshold)

@app.route('/reboot', methods=['POST'])
def reboot():
    subprocess.Popen(['sudo', 'reboot'])
    return redirect(url_for('index'))

@app.route('/confidence', methods=['POST'])
def set_confidence():
    global conf_threshold
    try:
        conf_threshold = float(request.form['confidence'])

        if conf_threshold < 0 or conf_threshold > 1:
            flash("Confidence value must be between 0.0 and 1.0", "error")
            return redirect(url_for('control'))

        flash(f"Confidence value accepted: {conf_threshold}", "success")
        return redirect(url_for('control'))
    except ValueError:
        flash("Invalid confidence value", "error")
        return redirect(url_for('control'))

def generate_camera():
    while True:
        frame = picam2.capture_array()
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def get_results():
    with lock:
        return jsonify(results_data)


@app.route('/temperature')
def temperature():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_str = f.readline()
            temp_c = round(int(temp_str) / 1000.0, 1)
            return jsonify({"temperature": temp_c})
    except Exception as e:
        return jsonify({"error": "Could not read temperature", "details": str(e)}), 500

if __name__ == '__main__':
    conf_threshold = 0.5  # Set your confidence threshold here
    t = threading.Thread(target=detect_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False)



def read_cpu_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            raw = f.read().strip()
            return float(raw) / 1000.0
    except Exception as e:
        print(f"[Error] Could not read CPU temp: {e}")
        return None

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'cpu_temperature_c'])

def log_cpu_temp():
    temp = read_cpu_temp_c()
    if temp is not None:
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), round(temp, 2)])
        print(f"[Log] CPU Temp: {temp:.2f}°C at {datetime.now().isoformat()}")