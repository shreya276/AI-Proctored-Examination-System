import cv2
import mediapipe as mp
import numpy as np
import threading

class HeadPoseDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cap = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.x = 0
        self.y = 0
        self.X_AXIS_CHEAT = 0
        self.Y_AXIS_CHEAT = 0

    def detect_head_pose(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                break

            # Flip the image horizontally to correct the inversion
            image = cv2.flip(image, 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.face_mesh.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []
            face_ids = [33, 263, 1, 61, 291, 199]

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None)
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in face_ids:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)
                    focal_length = 1 * img_w
                    cam_matrix = np.array([
                        [focal_length, 0, img_h / 2],
                        [0, focal_length, img_w / 2],
                        [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    self.x = angles[0] * 360
                    self.y = angles[1] * 360

                    # Determine head pose direction
                    if self.y < -10:
                        text = "You are Looking Left !!"
                        self.X_AXIS_CHEAT = 1
                    elif self.y > 10:
                        text = "You are Looking Right !!"
                        self.X_AXIS_CHEAT = 1
                    elif self.x < -10:
                        text = "You are Looking Down !!"
                        self.Y_AXIS_CHEAT = 1
                    else:
                        text = "Forward"
                        self.X_AXIS_CHEAT = 0
                        self.Y_AXIS_CHEAT = 0

                    text = f"{int(self.x)}::{int(self.y)} {text}"
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                    cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Head Pose Estimation', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def get_head_pose(self):
        return self.x, self.y

    def get_cheat_flags(self):
        return self.X_AXIS_CHEAT, self.Y_AXIS_CHEAT
