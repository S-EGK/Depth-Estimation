import cv2
import time
import torch
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline

# Function for stereo vision and depth estimation
import triangulation as tri
# import calibration

# load the mediapipe class for pose estimation
mp_pose_left = mp.solutions.pose # type: ignore
mp_pose_right = mp.solutions.pose # type: ignore
pose_left = mp_pose_left.Pose(static_image_mode=False)
pose_right = mp_pose_left.Pose(static_image_mode=False)

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

alpha1 = 0.2
previous_depth = 0.0
depth_scale = 1.0

#Applying exponential moving average filter
def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha1 * current_depth + (1 - alpha1) * previous_depth
    previous_depth = filtered_depth  # Update the previous depth value
    return filtered_depth

# Open both cameras
cap_right = cv2.VideoCapture(1, cv2.CAP_DSHOW)                    
cap_left =  cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 7               #Distance between the cameras [cm]
f = 3.67              #Camera lense's focal length [mm]
alpha = 78        #Camera field of view in the horisontal plane [degrees]

# Main program loop 
while True:

    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()
    img = frame_left.copy()

################## CALIBRATION #########################################################

    # frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)

########################################################################################

    start = time.time()
    
    # Convert the BGR image to RGB
    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and find poses
    results_right = pose_right.process(frame_right)
    results_left = pose_left.process(frame_left)

    # Check if landmarks are detected
    if results_left.pose_landmarks is not None and results_right.pose_landmarks is not None:
        # draw landmarks
        mp_drawing = mp.solutions.drawing_utils # type: ignore
        mp_drawing.draw_landmarks(frame_left, results_left.pose_landmarks, mp_pose_left.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame_right, results_right.pose_landmarks, mp_pose_right.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results_left.pose_landmarks, mp_pose_left.POSE_CONNECTIONS)

        # extract landmark coordinates
        landmarks_left = []
        landmarks_right = []
        for landmark_left, landmark_right in zip(results_left.pose_landmarks.landmark, results_right.pose_landmarks.landmark):
            landmarks_left.append((landmark_left.x, landmark_left.y, landmark_left.z))
            landmarks_right.append((landmark_right.x, landmark_right.y, landmark_right.z))

        shoulder_landmarks_left = [results_left.pose_landmarks.landmark[mp_pose_left.PoseLandmark.LEFT_SHOULDER],
                                    results_left.pose_landmarks.landmark[mp_pose_left.PoseLandmark.RIGHT_SHOULDER]]
        shoulder_landmarks_right = [results_right.pose_landmarks.landmark[mp_pose_right.PoseLandmark.LEFT_SHOULDER],
                                    results_right.pose_landmarks.landmark[mp_pose_right.PoseLandmark.RIGHT_SHOULDER]]
        
        mid_point_left = ((shoulder_landmarks_left[0].x + shoulder_landmarks_left[1].x)/2,
                            (shoulder_landmarks_left[0].y + shoulder_landmarks_left[1].y)/2,
                            (shoulder_landmarks_left[0].z + shoulder_landmarks_left[1].z)/2)
        mid_point_right = ((shoulder_landmarks_right[0].x + shoulder_landmarks_right[1].x)/2,
                            (shoulder_landmarks_right[0].y + shoulder_landmarks_right[1].y)/2,
                            (shoulder_landmarks_right[0].z + shoulder_landmarks_right[1].z)/2)
        mid_x_left, mid_y_left, mid_z_left = mid_point_left
        mid_x_right, mid_y_right, mid_z_right = mid_point_right

    # Apply input transforms for midas
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
    depth_mid_filt = spline(mid_y_left,mid_x_left) # type: ignore
    depth_midas = depth_to_distance(depth_mid_filt, depth_scale)
    depth_mid_filt = 100*(apply_ema_filter(depth_midas)/10)[0][0]

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    # convert color scheme
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert the RGB image to BGR
    frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
    frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)

    ################## CALCULATING DEPTH #########################################################

    # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
    # All formulas used to find depth is in video presentaion
    depth = tri.find_depth((mid_x_right*frame_right.shape[0],mid_y_right*frame_right.shape[1]), (mid_x_left*frame_left.shape[0],mid_y_left*frame_left.shape[1]), frame_right, frame_left, B, f, alpha) # type: ignore

    cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1)
    cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),1)
    # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
    # print("Depth: ", str(round(depth,1)))

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime
    
    cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.putText(img, f"FPS: {int(fps)}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
    cv2.putText(depth_map, "Estimated Depth: " + str(np.format_float_positional(depth_mid_filt, precision=1)) , (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1) # type: ignore

    # Resize and Show the frames
    frame_left = cv2.resize(frame_left, (500,350))
    frame_right = cv2.resize(frame_right, (500,350))
    img = cv2.resize(img, (500,350))
    depth_map = cv2.resize(depth_map, (500,350))
    cv2.imshow("frame right", frame_right) 
    cv2.imshow("frame left", frame_left)
    cv2.imshow("Image", img)
    cv2.imshow("Depth Map", depth_map)

    percentage_error = 100*abs(depth_mid_filt - depth)/(depth)
    print(percentage_error)

    # Hit "Esc" to close the window
    if cv2.waitKey(5) & 0xFF == 27:
        break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
