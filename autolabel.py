from ultralytics import YOLO
import os
import cv2
import shutil

model = YOLO('your_model.pt') #model

image_folder = 'images' #images path

output_folder_detected = 'output_detected'
os.makedirs(output_folder_detected, exist_ok=True)

output_folder_undetected = 'output_undetected'
os.makedirs(output_folder_undetected, exist_ok=True)

output_folder_txt_detected = 'output_txt_detected'
os.makedirs(output_folder_txt_detected, exist_ok=True)

output_folder_txt_undetected = 'output_txt_undetected'
os.makedirs(output_folder_txt_undetected, exist_ok=True)

for image_name in os.listdir(image_folder):
    if image_name.endswith('.png') or image_name.endswith('.jpg'):
        image_path = os.path.join(image_folder, image_name)

        results_list = model(image_path)

        if any(hasattr(results.boxes, 'xyxy') and len(results.boxes.xyxy) > 0 for results in results_list):

            detected_image_path = os.path.join(output_folder_detected, image_name)
            detected_txt_path = os.path.join(output_folder_txt_detected, os.path.splitext(image_name)[0] + '.txt')

            shutil.copy(image_path, detected_image_path)

            with open(detected_txt_path, 'w') as txt_file:
                original_image = cv2.imread(image_path)
                annotated_image = original_image.copy()

                for results in results_list:
                    if hasattr(results.boxes, 'xyxy'):
                        for box in results.boxes.xyxy:
                            box_coordinates = box.tolist()
                            txt_file.write(f"{box_coordinates[0]} {box_coordinates[1]} {box_coordinates[2]} {box_coordinates[3]}\n")
                            print(f"Coordinates: {box_coordinates}")

                            box_coordinates = [int(coord) for coord in box_coordinates]
                            cv2.rectangle(annotated_image, (box_coordinates[0], box_coordinates[1]),
                                          (box_coordinates[2], box_coordinates[3]), (0, 255, 0), 2)

                annotated_image_path = os.path.join(output_folder_detected, f"{image_name}")
                cv2.imwrite(annotated_image_path, annotated_image)

        else:
            undetected_image_path = os.path.join(output_folder_undetected, image_name)
            undetected_txt_path = os.path.join(output_folder_txt_undetected, os.path.splitext(image_name)[0] + '.txt')

            shutil.copy(image_path, undetected_image_path)

            with open(undetected_txt_path, 'w') as txt_file:
                txt_file.write("Undetected")

print("Process completed.")
