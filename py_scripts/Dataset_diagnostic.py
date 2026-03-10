import os
import cv2

# --- CONFIG ---
IMAGE_FOLDER = "SNN_Gate_Dataset/train/images" 
LABEL_FOLDER = "SNN_Gate_Dataset/train/labels" 

def check_txt_labels():
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith('.png') or f.endswith('.jpg')]
    
    for img_name in images:
        img_path = os.path.join(IMAGE_FOLDER, img_name)
        # Match image name to .txt name
        label_path = os.path.join(LABEL_FOLDER, os.path.splitext(img_name)[0] + ".txt")
        
        if not os.path.exists(label_path):
            print(f"Missing label for {img_name}")
            continue

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                if len(parts) < 5: continue
                
                # YOLO Format: class, x_center, y_center, width, height
                _, x_c, y_c, bw, bh = map(float, parts)

                # Convert to pixel coordinates
                start_x = int((x_c - bw/2) * w)
                start_y = int((y_c - bh/2) * h)
                end_x = int((x_c + bw/2) * w)
                end_y = int((y_c + bh/2) * h)

                # Draw the box the SNN will see
                cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        cv2.imshow("Label Checker", img)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_txt_labels()