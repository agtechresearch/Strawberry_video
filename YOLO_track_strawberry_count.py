#%%

# /home/cv_task/ultralytics/240516_ALARAD_1014_best.pt

# /home/cv_task/ultralytics/20221128_142439.mp4


#%%

import cv2

from ultralytics import YOLO, solutions

model = YOLO("/home/cv_task/ultralytics/240516_ALARAD_1014_best.pt") ################ weight file path
cap = cv2.VideoCapture("/home/cv_task/ultralytics/20221128_142439.mp4") ############# source path
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define line points
region_points = [(0, 00), (2000, 00), (2000,1500),(0, 1500)] ######################## region criteria

# Video writer ( output path ) #######################
video_writer = cv2.VideoWriter("/home/cv_task/ultralytics/vid_results/object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=True, #center point tail~~ 
    view_out_counts=False,
    line_thickness=2,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False, conf=0.4,)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()