import os
import cv2
import face_recognition
import shutil
import threading
import numpy as np
import pyaudio
import speech_recognition as sr
from tkinter import filedialog, messagebox, Button, Label, Entry, Frame, Tk
from tkinter.ttk import Style
from ttkthemes import ThemedTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from head_pose_integration import HeadPoseDetector
import queue
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# Define file path for storing user credentials
USER_DATA_FILE = "users.txt"

# Load YOLO model for object detection
net = cv2.dnn.readNet("utils/yolov3.weights", "utils/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
class_names = []
with open("utils/coco.names", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Function to register a new user
def register_user(username, password):
    if username_exists(username):
        return False
    with open(USER_DATA_FILE, "a") as user_file:
        user_file.write(f"{username},{password}\n")
    return True

def username_exists(username):
    if not os.path.exists(USER_DATA_FILE):
        return False
    with open(USER_DATA_FILE, "r") as user_file:
        for line in user_file:
            if line.strip().split(",")[0] == username:
                return True
    return False

def authenticate_user(username, password):
    if not os.path.exists(USER_DATA_FILE):
        return False
    with open(USER_DATA_FILE, "r") as user_file:
        for line in user_file:
            stored_username, stored_password = line.strip().split(",")
            if stored_username == username and stored_password == password:
                return True
    return False

def upload_image(username):
    image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", ".jpg;.png")])
    if image_path:
        registered_images_dir = "utils/registered_images"
        os.makedirs(registered_images_dir, exist_ok=True)
        new_image_path = os.path.join(registered_images_dir, f"{username}.jpg")
        shutil.copy(image_path, new_image_path)
        return new_image_path
    else:
        return None

def recognize_face(known_face_path):
    known_image = face_recognition.load_image_file(known_face_path)
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    # Start capturing video
    video_capture = cv2.VideoCapture(0)

    # Capture frames for 5 seconds
    start_time = cv2.getTickCount()
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Show bounding boxes for detected faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
            if True in matches:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, "Face Matched!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "No Match!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Webcam', frame)

        # Check if 5 seconds have passed
        if (cv2.getTickCount() - start_time) / cv2.getTickFrequency() > 5:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return face_locations, face_encodings, known_face_encoding

class ProctoringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Proctored Exam System")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Apply a modern theme
        self.style = Style()
        self.style.theme_use("clam")

        # Custom fonts
        self.title_font = ("Helvetica", 24, "bold")
        self.button_font = ("Helvetica", 14)
        self.label_font = ("Helvetica", 12)

        # Main frame
        self.main_frame = Frame(self.root, bg="#f0f0f0")
        self.main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Welcome label
        self.welcome_label = Label(self.main_frame, text="Welcome to AI Proctored Examination System", font=self.title_font, bg="#f0f0f0", fg="#333333")
        self.welcome_label.pack(pady=20)

        # Buttons
        self.register_button = Button(self.main_frame, text="Register", font=self.button_font, bg="#4CAF50", fg="white", command=self.open_register_window)
        self.register_button.pack(pady=10, ipadx=20, ipady=10)

        self.login_button = Button(self.main_frame, text="Login", font=self.button_font, bg="#2196F3", fg="white", command=self.open_login_window)
        self.login_button.pack(pady=10, ipadx=20, ipady=10)

        # Queue for thread-safe communication
        self.queue = queue.Queue()

    def open_register_window(self):
        self.register_window = Tk()
        self.register_window.title("Register")
        self.register_window.geometry("400x300")
        self.register_window.configure(bg="#f0f0f0")

        Label(self.register_window, text="Username", font=self.label_font, bg="#f0f0f0", fg="#333333").pack(pady=10)
        self.username_entry = Entry(self.register_window, font=self.label_font)
        self.username_entry.pack(pady=10)

        Label(self.register_window, text="Password", font=self.label_font, bg="#f0f0f0", fg="#333333").pack(pady=10)
        self.password_entry = Entry(self.register_window, show="*", font=self.label_font)
        self.password_entry.pack(pady=10)

        Button(self.register_window, text="Upload Image", font=self.button_font, bg="#FF9800", fg="white", command=self.upload_image).pack(pady=10)
        Button(self.register_window, text="Register", font=self.button_font, bg="#4CAF50", fg="white", command=self.register).pack(pady=10)

    def upload_image(self):
        username = self.username_entry.get()
        if username:
            self.image_path = upload_image(username)
        else:
            messagebox.showwarning("Warning", "Please enter a username before uploading an image.")

    def register(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        if register_user(username, password):
            messagebox.showinfo("Success", "User registered successfully!")
            self.register_window.destroy()
        else:
            messagebox.showerror("Error", "Username already exists!")

    def open_login_window(self):
        self.login_window = Tk()
        self.login_window.title("Login")
        self.login_window.geometry("400x300")
        self.login_window.configure(bg="#f0f0f0")

        Label(self.login_window, text="Username", font=self.label_font, bg="#f0f0f0", fg="#333333").pack(pady=10)
        self.login_username_entry = Entry(self.login_window, font=self.label_font)
        self.login_username_entry.pack(pady=10)

        Label(self.login_window, text="Password", font=self.label_font, bg="#f0f0f0", fg="#333333").pack(pady=10)
        self.login_password_entry = Entry(self.login_window, show="*", font=self.label_font)
        self.login_password_entry.pack(pady=10)

        Button(self.login_window, text="Login", font=self.button_font, bg="#2196F3", fg="white", command=self.login).pack(pady=10)

    def login(self):
        username = self.login_username_entry.get()
        password = self.login_password_entry.get()

        if authenticate_user(username, password):
            messagebox.showinfo("Success", "Authentication successful!")
            self.login_window.destroy()
            self.show_proctoring_dashboard(username)
        else:
            messagebox.showerror("Error", "Authentication failed! User not found.")

    def show_proctoring_dashboard(self, username):
        self.dashboard_window = Tk()
        self.dashboard_window.title("Proctoring Dashboard")
        self.dashboard_window.geometry("800x600")
        self.dashboard_window.configure(bg="#f0f0f0")

        Label(self.dashboard_window, text=f"Welcome, {username}", font=self.title_font, bg="#f0f0f0", fg="#333333").pack(pady=20)

        # Start Proctoring Button
        self.start_proctoring_button = Button(self.dashboard_window, text="Start Proctoring", font=self.button_font, bg="#4CAF50", fg="white", command=lambda: self.start_proctoring(username))
        self.start_proctoring_button.pack(pady=20, ipadx=20, ipady=10)

        # Initialize the graph
        self.initialize_graph()

    def initialize_graph(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title("Cheating Percentage")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Cheating Percentage")
        self.line, = self.ax.plot([], [], 'b-')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.dashboard_window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.cheating_data = []

    def update_graph(self, cheating_percentage):
        print(f"Updating graph with cheating percentage: {cheating_percentage}")  # Debug statement
        self.cheating_data.append(cheating_percentage)
        if len(self.cheating_data) > 100:
            self.cheating_data.pop(0)
        self.line.set_data(range(len(self.cheating_data)), self.cheating_data)
        self.ax.set_xlim(0, len(self.cheating_data))
        self.ax.set_ylim(0, 100)
        self.canvas.draw()

    def start_proctoring(self, username):
        known_face_path = f"utils/registered_images/{username}.jpg"
        if os.path.exists(known_face_path):
            face_locations, face_encodings, known_face_encoding = recognize_face(known_face_path)
            if face_encodings:
                messagebox.showinfo("Success", "Face matched! Starting proctoring...")
                self.webcam_monitoring(username, face_locations, known_face_encoding)
                self.start_voice_detection(username)
            else:
                messagebox.showerror("Error", "Face did not match. Access denied.")
        else:
            messagebox.showerror("Error", "No registered face image found for this user.")

    def webcam_monitoring(self, username, initial_face_locations, known_face_encoding):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        cap.set(cv2.CAP_PROP_FPS, 15)

        self.suspicious_activity_count = 0
        self.limit = 10
        self.frame_counter = 0

        head_pose_detector = HeadPoseDetector()
        head_pose_thread = threading.Thread(target=head_pose_detector.detect_head_pose, daemon=True)
        head_pose_thread.start()

        def process_frames(face_locations, known_face_encoding):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_counter += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
                    if True in matches:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, "Face Matched!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, "No Match!", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                x, y = head_pose_detector.get_head_pose()
                cv2.putText(frame, f"Head Pose: X: {int(x)} Y: {int(y)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if self.frame_counter % 5 == 0:
                    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                    net.setInput(blob)
                    outs = net.forward(output_layers)

                    class_ids = []
                    confidences = []
                    boxes = []

                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.5:
                                center_x = int(detection[0] * frame.shape[1])
                                center_y = int(detection[1] * frame.shape[0])
                                w = int(detection[2] * frame.shape[1])
                                h = int(detection[3] * frame.shape[0])

                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)

                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)

                    # Apply Non-Maximum Suppression
                    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

                    # Handle the case where indexes is a tuple
                    if isinstance(indexes, tuple):
                        indexes = np.array(indexes)
                    elif len(indexes) > 0:
                        indexes = indexes.flatten()

                    # Draw bounding boxes and labels
                    for i in indexes:
                        box = boxes[i]
                        (x, y, w, h) = box
                        label = str(class_names[class_ids[i]])
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        if label == "cell phone":
                            self.suspicious_activity_count += 1
                            self.log_suspicious_activity(username, "Cell phone detected")
                            cv2.putText(frame, "Phone Detected!", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if len(face_locations) > 1:
                    self.suspicious_activity_count += 1
                    self.log_suspicious_activity(username, "Multiple faces detected")
                    cv2.putText(frame, "Multiple Faces Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Proctoring", frame)

                if self.suspicious_activity_count >= self.limit:
                    self.log_out(username)
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            head_pose_thread.join()

        threading.Thread(target=process_frames, args=(initial_face_locations, known_face_encoding)).start()

    def start_voice_detection(self, username):
        def detect_voice():
            recognizer = sr.Recognizer()
            mic = sr.Microphone()

            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                while True:
                    audio = recognizer.listen(source)
                    try:
                        text = recognizer.recognize_google(audio)
                        if text:
                            self.queue.put(("voice", text))
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError:
                        pass

        threading.Thread(target=detect_voice, daemon=True).start()
        self.process_queue(username)

    def process_queue(self, username):
        while True:
            try:
                event, data = self.queue.get_nowait()
                if event == "voice":
                    self.suspicious_activity_count += 1
                    self.log_suspicious_activity(username, "Voice detected")
                    print("Voice detected:", data)
                if self.suspicious_activity_count >= self.limit:
                    self.log_out(username)
                    break
            except queue.Empty:
                pass

    def log_out(self, username):
        # Log out the user
        messagebox.showwarning("Warning", "Suspicious activity limit reached! Logging out...")
        self.dashboard_window.destroy()  # Close the proctoring dashboard

        # Generate and display ROC and Precision-Recall curves
        self.show_analysis_curves(username)

    def show_analysis_curves(self, username):
        # Load logged data
        try:
            data = pd.read_csv("malicious_activity_log.txt", names=["timestamp", "activity", "label"], sep=": ")
            data["is_true_positive"] = data["label"].apply(lambda x: 1 if "True Positive" in x else 0)
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        # Extract true labels and predicted scores (example scoring)
        y_true = data["is_true_positive"]
        y_scores = data["activity"].apply(lambda x: 1 if "detected" in x.lower() else 0)  # Example scoring

        # Compute Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Create a new window to display the curves
        curves_window = Tk()
        curves_window.title("Proctoring Analysis")
        curves_window.geometry("800x600")

        # Plot Precision-Recall curve
        fig1, ax1 = plt.subplots()
        ax1.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curve')
        ax1.legend()
        ax1.grid(True)

        canvas1 = FigureCanvasTkAgg(fig1, master=curves_window)
        canvas1.draw()
        canvas1.get_tk_widget().pack(pady=10)

        # Plot ROC curve
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True)

        canvas2 = FigureCanvasTkAgg(fig2, master=curves_window)
        canvas2.draw()
        canvas2.get_tk_widget().pack(pady=10)

        # Start the Tkinter event loop for the curves window
        curves_window.mainloop()

    def log_suspicious_activity(self, username, message):
        try:
            with open("malicious_activity_log.txt", "a") as log_file:
                log_file.write(f"{username}: {message}\n")
            print(f"Logged: {username}: {message}")  # Debug statement

            # Calculate cheating percentage
            cheating_percentage = (self.suspicious_activity_count / self.limit) * 100
            print(f"Cheating Percentage: {cheating_percentage}%")  # Debug statement
            self.update_graph(cheating_percentage)

            # Check if the limit is reached
            if self.suspicious_activity_count >= self.limit:
                self.log_out(username)  # Log out and show curves

        except Exception as e:
            print(f"Error writing to log file: {e}")

if __name__ == "__main__":
    root = ThemedTk(theme="arc")  # Use a modern theme
    app = ProctoringApp(root)
    root.mainloop()