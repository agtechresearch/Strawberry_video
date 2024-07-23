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
    reg_pts=region_points, # area
    classes_names=model.names,
    draw_tracks=True, #box center point tail~~ 
    view_out_counts=False,
    line_thickness=2,
)

class_names = ["Bud","Flower","Receptacle","Early fruit","White fruit","50% maturity", "80% maturity"]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(frame, persist=True, show=False, conf=0.4, verbose=True)

    # (현재 프레임용)탐지된 객체의 수 클래스별 집계
    class_counts = {class_name: 0 for class_name in class_names}
            
    if tracks[0].boxes.id is not None:
        for result in tracks[0]:
            for obj in result.boxes.data.tolist():
                class_id = int(obj[-1])
                conf = obj[5]
                
                class_name = class_names[class_id]
                class_counts[class_name] += 1
                            
    # 탐지된 객체 수를 프레임에 그리기(top left)
    y_offset = 20
    cv2.putText(frame, "Current", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_offset += 30
    for class_name, count in class_counts.items():
        if class_name == "Bud": count=class_counts["Flower"]
        elif class_name == "Flower": count=class_counts["Bud"]
        text = f"{class_name}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_offset += 30

    frame = counter.start_counting(frame, tracks) # counting result (Top right)
    video_writer.write(frame)



cap.release()
video_writer.release()
cv2.destroyAllWindows()