# Strawberry_video
### 레포지토리 내 파일 사용법 <br>
1. YOLOv8의 레포지토리 git clone진행
2. YOLOv8파일들 (ultralytics폴더) 안에 Strawberry_video의 파일들 위치 <br>
* 예시 (strawberry_counter.py, YOLO_track_strawberry_count.py 사용시)    
    > home <br>
    >> ultralytics  (git clone으로 생긴 파일들) <br>
    >>> -- .github <br>
    >>> -- docker <br>
    >>> -- docs <br>
    >>> -- examples <br>
    >>> -- ultralytics <br>
    >>> -- __strawberry_counter.py__ <br>
    >>> -- __YOLO_track_strawberry_count.py__




## YOLO_track_strawberry_count.py
* __기능__
    - YOLOv8기반 객체 탐지 및 tracking 결과 영상 생성
    - 프레임 좌측 상단: 현재 프레임 내 탐지결과
    - 우측 상단: 영상 내 누적 현황 
    - strawberry_counter와 함께 활용

* __주요 parameter__
    - source_path : input영상 path
    - weight_path : YOLOv8의 가중치 파일(.pt)
    - conf_v : YOLOv8 탐지에서 사용할 confidence 값
    - iou_v : YOLOv8 탐지에서 사용할 IoU값
    - class_names : class 이름 list
    - counter 설정
        - draw_tracks : 탐지된 객체들의 궤적을 보이게 할지(bool) 
        - view_counts : 오른쪽 상단 카운트 결과 보이게 할지(bool)
        - flag_point : 영상 내 몇번 탐지되었을 때부터 카운트할지
            - (순간적으로 탐지되는 객체의 경우 잘못 탐지된 경우가 많아 방지하기 위함, 영상의 이동속도가 빠르면 낮추고 느리면 높일 수 있음. 대부분 1초 30-40frame기준이기 때문에 30설정 시 1초이상 등장시 카운트함을 의미)

## strawberry_counter.py
* __기능__
    - YOLO_track_strawberry_count.py와 함께 활용
    - custom counter
    - 현재 화면상 보이는 객체를 카운트 하는 기능
    - flag_point를 이용해 몇회 이상 탐지되었을 때 카운트 할지 결정 가능
    - 탐지된 객체를 box와 함께 화면에 표시하는 기능을 포함

## video difference concat analyse.py
* __기능__
    - 특정 파라미터 및 방법론이 실제로 효과가 있었는지 확인하기 위해 작성
    - video path 1과 2가 비교할 두개의 영상이며
    - 두 영상을 활용하여 동일 시점의 프레임간 차이를 구해 시각적으로 표시하고 영상화 한 것이 video 3
    - 만약 아무 차이가 없으면 difference 계산 결과 영상에서 파란색 단색 화면만 보인다.
    - 최종적으로 비교를 위해 
        - video1 
        - video2
        - video3 순서로 concat되며 결과 영상으로 출력 

* __주요 parameter__
    - video_path1, video_path2 : 비교할 영상의 path
    - video_path3 : video1과 2의 차이를 계산하여 heatmap으로 시각화한 영상의 output path
    - output_path : concat영상의 output path
    - video_concat 함수
        - 첫번째 parameter의 경우 0과 1 중 사용하며
        - 0은 video1, 2를 단순히 concat한 영상만 필요할 때
        - 1은 두 영상의 차이를 시각화한 영상까지 concat한 영상을 출력


## strawberry_seg_bbox.py
별도의 segmentation algorithm 훈련 없이 기존 object detection모델을 활용하기 위해 작성

FastSAM 모델과 결합됨

## video_anti_shaker.py
영상의 흔들림을 줄여보기 위해 특정 영역을 가운데 두고 영상의 일부를 잘라서 흔들리는 문제를 줄이고자 작성

현재 기준은 재배베드의 검은색 비닐영역에 해당


## Simple_strawberry_count.ipynb
프레임당(이미지당) 탐지 결과를 활용하여 단순하게 계수하고 화면에 보여주기 위해 작성

마지막에 약 3초간 누적 결과(track id별 중복제거x)를 보여주도록 작성됨

사용하기 위해서는 id부여를 통한 동일객체 중복 집계 방지 등의 작업 필요
(사용x)




