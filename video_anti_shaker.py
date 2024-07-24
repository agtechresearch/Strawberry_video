import cv2
import numpy as np

# 동영상 파일 경로
input_video_path = '/home/cv_task/ultralytics/20221128_142439.mp4'
output_video_path = '/home/cv_task/ultralytics/stable_output_video.mp4'
#output_images_folder="/home/cv_task/ultralytics/crop_test_frame"

# 동영상 캡처
cap = cv2.VideoCapture(input_video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, 800))

i=0
smooth_y=[]
last_y_center=None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width = frame.shape[:2]


    # 피사체(검정색 비닐) 검출 - HSV 색상 공간을 이용
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([100, 40, 70])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # 모폴로지 연산으로 마스크 매끄럽게 만들기
    kernel1 = np.ones((15, 10), np.uint8)  # 커널 크기와 형태를 정의
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
    kernel2 = np.ones((8, 120), np.uint8)  # 커널 크기와 형태를 정의
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    
    desired_height=800
    # 검출된 영역의 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
            # 가장 큰 컨투어 선택
        c = max(contours, key=cv2.contourArea)
        x, y,_, _ = cv2.boundingRect(c)

        # 피사체 검출 영역을 사각형으로 표시
        #output_frame = frame.copy()
        """cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 검출된 피사체 영역을 이미지로 저장
        crop_img = frame[y:y + h, x:x + w]

        # 마스크 이미지 저장
        mask_image_path = os.path.join(output_images_folder, f'mask_.png')
        cv2.imwrite(mask_image_path, mask)

        """

    # 프레임 간 이동 제한
    if last_y_center is None:
        y_center = y
    else:
        y_center = last_y_center + (y - last_y_center) * 0.1  # 이동량을 10%로 제한

    smooth_y.append(y_center)
    y_center = np.mean(smooth_y[-4:])  # 최근 4개 프레임의 평균 사용
    
    y1=max(0,int(y_center-300))
    y2=min(height,int(y_center+(desired_height-y_center+y1)))
    last_y_center = y_center
    print("y1, y, y2:",y1,y,y2)

    # 크롭 및 프레임 조정
    cropped_frame = frame[y1:y2,0:width]
    
    # 피사체 영역 표시 이미지 저장
    #output_frame_image_path = os.path.join(output_images_folder, f'frame_{i}.png')
    #cv2.imwrite(output_frame_image_path, cropped_frame)
    i+=1

    if cropped_frame.shape[0] != desired_height :
        print(cropped_frame.shape)
        print("error")
        
    video_writer.write(cropped_frame)

   
# 자원 해제
cap.release()
video_writer.release()
cv2.destroyAllWindows()