import cv2
import face_recognition
import time
import threading
import voice_detection
from head_pose_integration import HeadPoseDetector

def recognize_face_with_preview(known_face_path, callback):
    known_image = face_recognition.load_image_file(known_face_path)
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)  # Width
    video_capture.set(4, 480)  # Height

    face_matched = False
    start_time = time.time()

    head_pose_detector = HeadPoseDetector()
    head_pose_thread = threading.Thread(target=head_pose_detector.detect_head_pose, daemon=True)
    head_pose_thread.start()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_matched = False
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
            if True in matches:
                face_matched = True
                break

        if face_matched:
            cv2.putText(frame, "Face Matched!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected! Try again!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        x, y = head_pose_detector.get_head_pose()
        cv2.putText(frame, f"Head Pose: X: {int(x)} Y: {int(y)}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Webcam', frame)

        if face_matched and (time.time() - start_time > 5):
            print("Face verified! Starting proctoring...")
            with open("malicious_activity_log.txt", "a") as log_file:
                log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Face Verified - Proctoring Started\n")

            voice_thread = threading.Thread(target=voice_detection.detect_voice, daemon=True)
            voice_thread.start()

            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    head_pose_thread.join()
    callback(face_matched)
