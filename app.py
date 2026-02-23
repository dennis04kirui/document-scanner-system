from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import cv2
import numpy as np
import os
from PIL import Image
import time
import sqlite3
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pytesseract
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ------------------- SETUP ------------------- #
app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
PDF_FOLDER = 'static/pdf/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# ------------------- USER MODEL ------------------- #
class User(UserMixin):
    def __init__(self, id_, username, password):
        self.id = id_
        self.username = username
        self.password = password

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

init_db()

def get_user_by_username(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return User(row[0], row[1], row[2])
    return None

def create_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = generate_password_hash(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return False
    conn.close()
    return True

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return User(row[0], row[1], row[2])
    return None

# ------------------- DOCUMENT SCANNER ------------------- #
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = pts
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_rotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        osd = pytesseract.image_to_osd(gray)
        rotation = int(osd.split("Rotate:")[1].split("\n")[0])
        if rotation != 0:
            (h, w) = image.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), -rotation, 1.0)
            image = cv2.warpAffine(image, M, (w, h),
                                   flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_REPLICATE)
    except:
        pass
    return image

def process_image(image_path):
    img = cv2.imread(image_path)
    img = auto_rotate(img)
    orig = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    min_val, max_val = np.min(gray), np.max(gray)
    if max_val != min_val:
        gray = cv2.convertScaleAbs(gray, alpha=255/(max_val - min_val), beta=-min_val*255/(max_val - min_val))
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 75, 200)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break

    warped = four_point_transform(orig, order_points(doc_cnt.reshape(4,2))) if doc_cnt is not None else orig
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.adaptiveThreshold(warped_gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    timestamp = int(time.time())
    filename = f"{current_user.username}_{timestamp}.png"
    processed_path = os.path.join(PROCESSED_FOLDER, filename)
    cv2.imwrite(processed_path, enhanced)

    # Remove the uploaded file after processing
    os.remove(image_path)
    return filename

# ------------------- DELETE ROUTES ------------------- #
@app.route('/delete/<filename>', methods=['POST'])
@login_required
def delete_file(filename):
    if not filename.startswith(current_user.username):
        return "Unauthorized", 403
    file_path = os.path.join(PROCESSED_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        flash("File deleted successfully.")
    else:
        flash("File not found.")
    return redirect(url_for('index'))

@app.route('/delete_all', methods=['POST'])
@login_required
def delete_all():
    deleted_count = 0
    for file in os.listdir(PROCESSED_FOLDER):
        if file.startswith(current_user.username):
            os.remove(os.path.join(PROCESSED_FOLDER, file))
            deleted_count += 1
    flash(f"{deleted_count} file(s) deleted.")
    return redirect(url_for('index'))

# ------------------- ROUTES ------------------- #
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = get_user_by_username(username)
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        flash("Invalid credentials")
    return render_template('login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if create_user(username, password):
            flash("Account created! Please login.")
            return redirect(url_for('login'))
        flash("Username already exists.")
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET','POST'])
@login_required
def index():
    processed_images = [f for f in os.listdir(PROCESSED_FOLDER) if f.startswith(current_user.username)]
    if request.method == 'POST':
        files = request.files.getlist('images')
        for file in files:
            if file.filename != '':
                safe_name = secure_filename(file.filename)
                upload_path = os.path.join(UPLOAD_FOLDER, f"{current_user.username}_{int(time.time())}_{safe_name}")
                file.save(upload_path)
                processed_images.append(process_image(upload_path))
    return render_template('index.html', processed_images=processed_images)

@app.route('/download_pdf')
@login_required
def download_pdf():
    images = []
    for img_file in os.listdir(PROCESSED_FOLDER):
        if img_file.startswith(current_user.username):
            img_path = os.path.join(PROCESSED_FOLDER, img_file)
            img = Image.open(img_path).convert('RGB')
            images.append(img)
    if not images:
        return "No processed images available.", 400
    pdf_path = os.path.join(PDF_FOLDER, f"{current_user.username}_scanned.pdf")
    images[0].save(pdf_path, save_all=True, append_images=images[1:])
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)