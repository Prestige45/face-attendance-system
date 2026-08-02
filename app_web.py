import os
import sqlite3
import cv2
import numpy as np
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for

app = Flask(__name__)
app.secret_key = "super_secure_btech_project_key" 

# --- ADMIN CREDENTIALS ---
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "password123" 

# --- 1. SYSTEM INITIALIZATION ---
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

def init_db():
    conn = sqlite3.connect('attendance_system.db')
    conn.execute("CREATE TABLE IF NOT EXISTS students (student_id TEXT PRIMARY KEY, name TEXT, matric_no TEXT UNIQUE)")
    conn.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, date TEXT, time TEXT, status TEXT)")
    conn.commit()
    conn.close()

init_db()

# --- 2. OPENCV AI ENGINE ---
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)
recognizer = cv2.face.LBPHFaceRecognizer_create()

def load_model():
    if os.path.exists("trainer.yml"):
        recognizer.read("trainer.yml")
        print("✅ AI Model Loaded Successfully")
        return True
    return False

load_model()

def perform_training():
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(".jpg")]
    face_samples, ids = [], []
    for image_path in image_paths:
        img_numpy = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        id_val = int(os.path.split(image_path)[-1].split(".")[1])
        face_samples.append(img_numpy)
        ids.append(id_val)
        
    if len(face_samples) > 0:
        recognizer.train(face_samples, np.array(ids))
        recognizer.write("trainer.yml")
        load_model()
        return True
    else:
        if os.path.exists("trainer.yml"):
            os.remove("trainer.yml")
        return False

def get_db_connection():
    conn = sqlite3.connect('attendance_system.db')
    conn.row_factory = sqlite3.Row
    return conn

# --- 3. WEB PAGES & SECURITY ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin'))
        else:
            return render_template('login.html', error="Invalid Authentication Credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('login'))

@app.route('/admin')
def admin():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    today = datetime.now().strftime("%Y-%m-%d")
    
    total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    present_today = conn.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,)).fetchone()[0]
    
    attendance = conn.execute('''
        SELECT a.date, a.time, s.name, s.matric_no, a.status 
        FROM attendance a JOIN students s ON a.student_id = s.student_id 
        ORDER BY a.date DESC, a.time DESC
    ''').fetchall()
    
    students = conn.execute("SELECT * FROM students ORDER BY CAST(student_id AS INTEGER) DESC").fetchall()
    conn.close()
    
    return render_template('admin.html', records=attendance, students=students, 
                           total_students=total_students, present_today=present_today)

@app.route('/dataset/<path:filename>')
def download_file(filename):
    if not session.get('admin_logged_in'):
        return "Unauthorized", 401
    return send_from_directory(dataset_path, filename)

# --- 4. API: STUDENT MANAGEMENT (CRUD) ---
@app.route('/api/update_student', methods=['POST'])
def update_student():
    if not session.get('admin_logged_in'): return jsonify({"status": "error", "message": "Unauthorized"})
    data = request.json
    conn = get_db_connection()
    try:
        conn.execute("UPDATE students SET name=?, matric_no=? WHERE student_id=?", (data['name'], data['matric'], data['student_id']))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "Record Updated"})
    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "Matric number already exists"})

@app.route('/api/delete_student/<student_id>', methods=['POST'])
def delete_student(student_id):
    if not session.get('admin_logged_in'): return jsonify({"status": "error", "message": "Unauthorized"})
    conn = get_db_connection()
    conn.execute("DELETE FROM students WHERE student_id=?", (student_id,))
    conn.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
    conn.commit()
    conn.close()
    for f in os.listdir(dataset_path):
        if f.startswith(f"user.{student_id}."): os.remove(os.path.join(dataset_path, f))
    perform_training()
    return jsonify({"status": "success"})

# --- 5. API: CORE ENGINE & SECURITY ---
@app.route('/api/init_student', methods=['POST'])
def init_student():
    data = request.json
    conn = get_db_connection()
    if conn.execute("SELECT student_id FROM students WHERE matric_no=?", (data['matric'],)).fetchone():
        return jsonify({"status": "error", "message": "Matric number already registered!"})
    max_id = conn.execute("SELECT MAX(CAST(student_id AS INTEGER)) FROM students").fetchone()[0]
    new_id = str(1 if max_id is None else max_id + 1)
    conn.execute("INSERT INTO students (student_id, name, matric_no) VALUES (?, ?, ?)", (new_id, data['name'], data['matric']))
    conn.commit()
    conn.close()
    return jsonify({"status": "success", "student_id": new_id})

@app.route('/api/save_frame', methods=['POST'])
def save_frame():
    data = request.json
    img = cv2.imdecode(np.frombuffer(base64.b64decode(data['image'].split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))
    
    if len(faces) > 0:
        x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face_roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100))

        # --- BIOMETRIC UNIQUENESS CHECK (Prevents Duplicate Faces) ---
        if data['count'] <= 3 and os.path.exists("trainer.yml"):
            try:
                predicted_id, conf = recognizer.predict(face_roi)
                if conf < 70: # AI is highly confident this face already exists!
                    conn = get_db_connection()
                    conn.execute("DELETE FROM students WHERE student_id=?", (str(data['student_id']),))
                    conn.commit()
                    conn.close()
                    return jsonify({"status": "duplicate", "message": "Face already registered!"})
            except:
                pass 

        cv2.imwrite(f"{dataset_path}/user.{data['student_id']}.{data['count']}.jpg", face_roi)
        return jsonify({"status": "success", "face_found": True})
        
    return jsonify({"status": "success", "face_found": False})

@app.route('/api/train', methods=['POST'])
def train_model():
    if perform_training(): return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Training failed"})

@app.route('/api/scan', methods=['POST'])
def scan_face():
    if not os.path.exists("trainer.yml"): return jsonify({"status": "error", "message": "System not trained."})
    img = cv2.imdecode(np.frombuffer(base64.b64decode(request.json['image'].split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(120, 120))
    if len(faces) == 0: return jsonify({"status": "scanning", "message": "Scanning for faces..."})
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    try:
        student_id, conf = recognizer.predict(cv2.resize(gray[y:y+h, x:x+w], (100, 100)))
        if conf < 85:
            conn = get_db_connection()
            student = conn.execute("SELECT name FROM students WHERE student_id=?", (str(student_id),)).fetchone()
            if student:
                name = student['name']
                today = datetime.now().strftime("%Y-%m-%d")
                if conn.execute("SELECT id FROM attendance WHERE student_id=? AND date=?", (str(student_id), today)).fetchone():
                    return jsonify({"status": "warning", "message": f"{name} - ALREADY CAPTURED"})
                conn.execute("INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)", (str(student_id), today, datetime.now().strftime("%H:%M:%S"), "Present"))
                conn.commit()
                return jsonify({"status": "success", "message": f"ATTENDANCE CAPTURED: {name}"})
    except: pass
    return jsonify({"status": "unknown", "message": "UNREGISTERED FACE DETECTED"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)