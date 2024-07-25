# Strawberry_video


## YOLO_track_strawberry_count.py
YOLOv8기반 객체 탐지 및 tracking 결과 영상 생성

프레임 좌측 상단: 현재 프레임 내 탐지결과

우측 상단: 특정 영역을 지나간 객체 누적 현황

## Simple_strawberry_count.ipynb
프레임당(이미지당) 탐지 결과를 활용하여 단순하게 계수하고 화면에 보여주기 위해 작성

마지막에 약 3초간 누적 결과를 보여주도록 작성됨

사용하기 위해서는 id부여를 통한 동일객체 중복 집계 방지 등의 작업 필요
(사용x)

## video difference concat analyse.py
특정 파라미터 및 방법론이 실제로 효과가 있었는지 확인하기 위해 작성

video path 1과 2가 비교할 두개의 영상이며

두 영상을 활용하여 동일 시점의 프레임간 차이를 구해 시각적으로 표시하고 영상화 한 것이 video 3

만약 아무 차이가 없으면 파란색 단색 화면만 보인다.

최종적으로 비교를 위해 
video1
video2
video3 순서로 concat되며 결과 영상으로 출력 

## strawberry_seg_bbox.py
별도의 segmentation algorithm 훈련 없이 기존 object detection모델을 활용하기 위해 작성

FastSAM 모델과 결합됨

## video_anti_shaker.py
영상의 흔들림을 줄여보기 위해 특정 영역을 가운데 두고 영상의 일부를 잘라서 흔들리는 문제를 줄이고자 작성

현재 기준은 재배베드의 검은색 비닐영역에 해당
