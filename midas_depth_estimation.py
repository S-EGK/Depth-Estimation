import cv2
import time
import torch
import numpy as np
import mediapipe as mp
from scipy.interpolate import RectBivariateSpline

# load the mediapipe class for pose estimation
mp_pose = mp.solutions.pose # type: ignore
pose = mp_pose.Pose(static_image_mode=False)

#%% Load a MiDas model for depth estimation
# Load a model (see https://github.com/intel-isl/MiDaS/#Accuracy for an overview)
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo="True")

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image for large or small model
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo="True")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# Converting depth to distance
def depth_to_distance(depth_value, depth_scale):
    return 1.0/(depth_value*depth_scale)

alpha = 0.2
previous_depth = 0.0
depth_scale = 1.0

#Applying exponential moving average filter
def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth  # Update the previous depth value
    print(previous_depth, filtered_depth)
    return filtered_depth

# Open up the video capture from a webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while cap.isOpened():
    # read image from camera
    success, img = cap.read()
    # to caculate time to process
    start = time.time()
    # convert color scheme
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detect the body landmarks in the frame
    results = pose.process(img)
    # check if landmarks are detected
    if results.pose_landmarks is not None:
        # draw landmarks
        mp_drawing = mp.solutions.drawing_utils # type: ignore
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract Landmark Coordinates
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))

        shoulder_landmarks = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                           results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]]

        mid_point = ((shoulder_landmarks[0].x + shoulder_landmarks[1].x) / 2,
                     (shoulder_landmarks[0].y + shoulder_landmarks[1].y) / 2,
                     (shoulder_landmarks[0].z + shoulder_landmarks[1].z) /2)
        mid_x, mid_y, mid_z = mid_point

    # Apply input transforms
    input_batch = transform(img).to(device)

    # prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # type: ignore

    # Creating a spline array of non-integer grid
    h, w = depth_map.shape
    x_grid = np.arange(w)
    y_grid = np.arange(h)

    # Create a spline object using the depth_map array
    spline = RectBivariateSpline(y_grid, x_grid, depth_map)
    depth_mid_filt = spline(mid_y,mid_x) # type: ignore
    depth_midas = depth_to_distance(depth_mid_filt, depth_scale)
    depth_mid_filt = (apply_ema_filter(depth_midas)/10)[0][0]

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    # convert color scheme
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize images
    img = cv2.resize(img, (500,350))
    depth_map = cv2.resize(depth_map, (500,350))

    # calculate fps
    end = time.time()
    total_time = end - start
    fps = 1/total_time

    cv2.putText(img, f"FPS: {int(fps)}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
    cv2.imshow("Image", img)
    cv2.putText(depth_map, "Estimated Depth: " + str(
            np.format_float_positional(depth_mid_filt*100 , precision=1)) , (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 1)
    cv2.imshow("Depth Map", depth_map)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()