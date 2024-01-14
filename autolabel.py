from ultralytics import YOLO
import os
import cv2

model = YOLO('model.pt') #model

image_folder = 'images'

output_folder_txt = 'output_txt'
os.makedirs(output_folder_txt, exist_ok=True)

output_folder_images = 'output_images'
os.makedirs(output_folder_images, exist_ok=True)

for image_name in os.listdir(image_folder):
    if image_name.endswith('.png') or image_name.endswith('.jpg'):
        image_path = os.path.join(image_folder, image_name)

        results_list = model(image_path)  
        txt_filename = os.path.splitext(image_name)[0] + '.txt'
        txt_path = os.path.join(output_folder_txt, txt_filename)
        with open(txt_path, 'w') as txt_file:
            for results in results_list:
                if hasattr(results.boxes, 'xyxy'):
                    for box in results.boxes.xyxy:
                        box_coordinates = box.tolist()
                        txt_file.write(f"{box_coordinates[0]} {box_coordinates[1]} {box_coordinates[2]} {box_coordinates[3]}\n")
                        print(f"Coordinates: {box_coordinates}")
                if hasattr(results, 'probs') and results.probs is not None:
                    probs = results.probs[0].tolist()
                    txt_file.write(f"{probs[0]}\n") 

        original_image = cv2.imread(image_path)
        annotated_image = original_image.copy()
        for results in results_list:
            if hasattr(results.boxes, 'xyxy'):
                for box in results.boxes.xyxy:
                    box_coordinates = box.tolist()
                    box_coordinates = [int(coord) for coord in box_coordinates]
                    cv2.rectangle(annotated_image, (box_coordinates[0], box_coordinates[1]),
                                  (box_coordinates[2], box_coordinates[3]), (0, 255, 0), 2)

        annotated_image_path = os.path.join(output_folder_images, f"annotated_{image_name}")
        cv2.imwrite(annotated_image_path, annotated_image)
