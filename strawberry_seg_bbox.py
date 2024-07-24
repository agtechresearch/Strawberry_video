import cv2
import numpy as np
from ultralytics import YOLO



def overlay(image, mask, color, alpha, bbox):
    
    x1, y1, x2, y2 = map(int,bbox)
    
    color = color[::-1]
    mask = mask.data
    colored_mask = np.stack([mask]*3, axis=-1).astype(np.uint8)
    color_mask = np.zeros_like(colored_mask, dtype=np.uint8)
    color_mask[mask == 1] = color
    color_mask = np.squeeze(color_mask)

   
    # Debugging: shape and dimension check
    print(f"Image shape: {image[int(y1):int(y2), int(x1):int(x2)].shape}")
    print(f"Color mask shape: {color_mask.shape}")
    
    image[int(y1):int(y2), int(x1):int(x2)] = cv2.addWeighted(image[int(y1):int(y2), int(x1):int(x2)], 1 - alpha, color_mask, alpha, 0)

    return image


# YOLOv8 load
model = YOLO('/home/cv_task/ultralytics/ALARAD_y8m_2023_2024_cont_best.pt')
# Load the original image
image = "/home/imagepath/20231209_131738_L2_V1_P17_D1050.png"
results = model(image)
img = cv2.imread(image)
# Extract bounding boxes
boxes = results[0].boxes.xyxy.tolist()
classes=results[0].boxes.cls.tolist()
# Iterate through the bounding boxes

crop_list=[]
for i, box in enumerate(boxes):
    if classes[i] >= 3:
      x1, y1, x2, y2 = box
      # Crop the object using the bounding box coordinates
      x1=x1-5
      x2=x2+5
      y1=y1-5
      y2=y2+5

      ultralytics_crop_object = img[int(y1):int(y2), int(x1):int(x2)]
      # Save the cropped object as an image
      #cv2.imwrite('/home/path/crop_' + str(i) + '.jpg', ultralytics_crop_object)

      crop_list.append([ultralytics_crop_object, [x1,y1,x2,y2],classes[i]])
      
#crop area segmentation

from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt

# Create a FastSAM model
model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt
# Run inference on an image
_,width=img.shape[:2]

result_list=[]
for i in range(len(crop_list)):
    height, width = crop_list[i][0].shape[:2]
    everything_results = model(crop_list[i][0], device="cpu", retina_masks=True, imgsz=height, conf=0.3, iou=0.5)
    # Prepare a Prompt Process object
    prompt_process = FastSAMPrompt(crop_list[i][0], everything_results, device="cpu")
    # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]

    center_x = width // 2
    center_y = height // 2

    results = prompt_process.point_prompt(points=[[center_x,center_y]],pointlabel=[1])
    #print(results[0])
    result_list.append([results[0].masks, crop_list[i][1]])

#prompt_process.plot(annotations=results, output="/home/cv_task/ultralytics/vid_results/result_"+str(i))

color=[255,0,0]
alpha=0.5
image_overlay = img.copy()
for mask, bbox in result_list:
    image_overlay = overlay(image_overlay, mask, color, alpha, bbox)

# Save the result
output_path = 'path/image_seg_overlay.jpg'
cv2.imwrite(output_path, image_overlay)
