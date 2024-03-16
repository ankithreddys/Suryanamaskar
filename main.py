import cv2
import torch
import time
import os
import numpy as np
from torchvision.io import read_image
from torchvision import transforms
from ultralytics import YOLO
from DATA_PROCESS.data_transformation import transformations


import warnings
warnings.filterwarnings('ignore')

def classification(model, person_image, device):
    print("classification starting")
    model.eval()
    data_trans = transformations()
    #img = read_image(path)
    img = transforms.ToPILImage()(person_image)
    img = data_trans['validation'](img).to(device)
    output = model.features(img)
    output = output.view(-1)
    output = model.classifier(output)

    predicted_class = np.argmax(output.detach().cpu().numpy())

    probs = torch.nn.functional.softmax(output)
    #max_prob = torch.max(probs)
    print(probs)
    '''if max_prob > 0.6:
        return predicted_class
    else:
        return -100
    '''
    return predicted_class
def main():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(device)

    class_model = torch.load("C:/Users/ankit/Music/suryanamaskar/MODELS/CLASSIFICATION_MODELS_SAVE_DIR/best_model.pt")
    #del class_model.avg_pool
    class_model.to(device=device)



    # Load a pre-trained YOLOv8 model
    yolo_model = YOLO('C:/Users/ankit/Music/suryanamaskar/MODELS/yolov8m.pt')
    yolo_model.to(device)

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Define a threshold for considering a person "static"
    static_threshold = 4
    # Define the minimum static duration in seconds
    static_duration = 2

    # Initialize variables for tracking movement
    previous_box = None
    previous_center = None
    static_start_time = None 
    i = 0

    while True:
        ret, frame = cap.read()

        results = yolo_model.predict(frame)
        #print(type(results[0].boxes.cls.cpu().numpy()))
        
        try:
        # Check if a person is detected
            if results[0].boxes.cls.cpu().numpy()[0] == 0:
                person_data = results[0].boxes
                # Get the person's bounding box coordinates
                person_box = person_data.xyxy.cpu().numpy()

                # Calculate the center of the bounding box
                center_x = (person_box[0][0] + person_box[0][2]) / 2
                center_y = (person_box[0][1] + person_box[0][3]) / 2

                # Track movement
                if previous_center is not None:
                    distance = abs(center_x - previous_center[0]) + abs(center_y - previous_center[1])

                    # Check if the person is static
                    if distance < static_threshold:
                        if static_start_time is None:
                            # Person is becoming static, start timer
                            static_start_time = time.time()
                        elif time.time() - static_start_time >= static_duration:
                            # Person has been static for long enough, save the image
                            #path = f"person_static_image_{i+1}.jpg"
                            #cv2.imwrite(f"person_static_image_{i+1}.jpg", frame)
                            person_image = frame[int(person_box[0][1]):int(person_box[0][3]),
                                                  int(person_box[0][0]):int(person_box[0][2])]
                            img_class = classification(model=class_model,person_image = person_image,device = device)
                            print("classification done")
                            cv2.putText(person_image,str(img_class), (50, 50), cv2.FONT_HERSHEY_SIMPLEX , 1,  (0, 255, 255),2, cv2.LINE_4)
                            #os.chdir("C:/Users/ankit/Music/suryanamaskar/Recorded frames")
                            cv2.imwrite(f"person_static_image_{i+1}.jpg", person_image)

                            #os.chdir("C:/Users/ankit/Music/suryanamaskar")
                            i += 1
                            static_start_time = None
                    else:
                        # Person moved, reset static timer
                        static_start_time = None

                # Update previous positions
                previous_box = person_box
                previous_center = (center_x, center_y)
            
        except:
            print('no image detected')
        # Display the frame
        cv2.imshow("Frame", frame)

        # Exit if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()