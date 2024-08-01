#%%

# /home/cv_task/ultralytics/240516_ALARAD_1014_best.pt

# /home/cv_task/ultralytics/20221128_142439.mp4


#%%

import cv2

from ultralytics import YOLO
from strawberry_counter import StrawberryCounter

source_path="/home/cv_task/ultralytics/stable_output_video.mp4" ############# source path
weight_path="/home/cv_task/ultralytics/240516_ALARAD_1014_best.pt" ################ weight file path

source_name=source_path.split('/')[-1].split('.')[0]
vid_output_path=f"/home/cv_task/ultralytics/vid_results/strawberry_count_{source_name}.avi"
conf_v=0.28 # confidenc value for YOLOv8
iou_v=0.6 # iou threshold for YOLOv8

model = YOLO(weight_path) 
cap = cv2.VideoCapture(source_path) 


assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define line points
region_points = [(0, 00), (w, 00), (w,h),(0, h)]  ######################## region criteria

# Video writer ( output path ) #######################
video_writer = cv2.VideoWriter(vid_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = StrawberryCounter(
    view_img=True,
    reg_pts=region_points,
    classes_names=model.names,
    draw_tracks=False, #center point tail~~ 
    view_counts=True,
    line_thickness=2,
    flag_point=20, #영상 내 n번 이상 탐지되면 카운트(순간적으로 등장시 카운트 방지)
)

class_names = ["Bud","Flower","Receptacle","Early fruit","White fruit","50% maturity", "80% maturity"]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(frame, persist=True, show=False,iou=iou_v, conf=conf_v, verbose=True, tracker="botsort.yaml")


    # (현재 프레임용)탐지된 객체의 수 클래스별 집계
    class_counts = {class_name: 0 for class_name in class_names}
            
    if tracks[0].boxes.id is not None:
        for idx,result in enumerate(tracks[0]):
            for obj in result.boxes.data:
                class_id = int(obj[-1])
                conf = obj[5]
                
                #x1,y1,x2,y2,id,conf,classes=obj

                class_name = class_names[class_id]
                class_counts[class_name] += 1   
                            
    # 탐지된 객체 수를 프레임에 그리기(top left)
    y_offset = 20
    cv2.putText(frame, "Current", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_offset += 30
    for class_name, count in class_counts.items():
        
        text = f"{class_name}: {count}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        y_offset += 30

    frame = counter.start_counting(frame, tracks) # 영역을 지난, 누적된 객체 수를 프레임에 그리기 (Top right)
    video_writer.write(frame)

counter.return_count_result_print()

cap.release()
video_writer.release()
cv2.destroyAllWindows()

print("saved at",vid_output_path)