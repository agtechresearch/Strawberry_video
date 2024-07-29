#%%
import cv2
import numpy as np

video_path1 = "/home/cv_task/ultralytics/vid_results/object_track_default_conf028.avi"
video_path2 = "/home/cv_task/ultralytics/vid_results/object_track_default_conf028_iou06.avi" 
video_path3 = '/home/cv_task/ultralytics/vid_results/difference_video.mp4'  
output_path = '/home/cv_task/ultralytics/vid_results/output_concat_analyze_video.mp4'  #final output

# 0=> video path1 & video path2만 concat
# 1=> video1, video2, difference 3개영상 concat
concat_mode=0

def cal_difference(video_path1, video_path2, output_video_path): #두 영상의 차이를 시각화한 동영상 생성
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    #warning!
    frame_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (frame_width != frame_width2) or (frame_height != frame_height2):
        print("video width height difference")

    # save video
    output_path = output_video_path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 비디오 코덱 (MP4)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # grayscale for calculate
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # diff image generate
        diff_image = cv2.absdiff(gray1, gray2)
        
        # diff image visualize
        heatmap = cv2.applyColorMap(diff_image, cv2.COLORMAP_JET)
        
        # save frame
        out.write(heatmap)

    # resource del
    cap1.release()
    cap2.release()
    out.release()
    print("video complete")

# 비교1영상, 비교2영상, 차이영상 수직으로 붙인 영상 생성
def video_concat(mode,out_path,path1,path2,path3=None):
    
    cap1 = cv2.VideoCapture(path1)
    cap2 = cv2.VideoCapture(path2)
    if mode == 1:
        cap3 = cv2.VideoCapture(video_path3)

    title1=path1.split('/')[-1]
    title2=path2.split('/')[-1]

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height * (2+int(mode))))

    f_idx=0
    temp=[]
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if mode == 1:
            ret3, frame3 = cap3.read()
        
        if not ret1 or not ret2:
            break
        
        # frame concat
        if mode == 0:
            combined_frame = cv2.vconcat([frame1, frame2])
        else:
            combined_frame = cv2.vconcat([frame1, frame2,frame3])

        # 텍스트 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (255, 255, 255)  # 흰색
        cv2.putText(combined_frame, title1, (50, 50), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(combined_frame, title2, (50, height + 50), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # save
        out.write(combined_frame)
        f_idx +=1

    # resource release
    cap1.release()
    cap2.release()
    if mode == 1 : 
        cap3.release()
    out.release()
    cv2.destroyAllWindows()

    print("Saved at",output_path)
    print("done")

cal_difference(video_path1,video_path2,video_path3)

video_concat(concat_mode, output_path, video_path1,video_path2,video_path3)