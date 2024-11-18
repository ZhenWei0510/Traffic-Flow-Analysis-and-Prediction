from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import requests

# discord webhook url
url = 'webhook url'

data = {
    "content" : "程式啟動",
}

response = requests.post(url, json=data)   # 使用 POST 方法

# 記錄上次發送通知的時間
last_hour = None

class_names = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
]

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = 'https://cctv4.kctmc.nat.gov.tw/9e00aa3e/' # 民族一路、大順二路
cap = cv2.VideoCapture(video_path)

# Variables to store tracking information and vehicle count
track_history = defaultdict(lambda: [])
counted_ids = set()  # Stores IDs of vehicles that have crossed the line
up_lane1_count = 0  # Counter for vehicles that cross the line
up_lane2_count = 0  # Counter for vehicles that cross the line
up_lane3_count = 0  # Counter for vehicles that cross the line
up_lane4_count = 0  # Counter for vehicles that cross the line
down_lane1_count = 0  # Counter for vehicles that cross the line
down_lane2_count = 0  # Counter for vehicles that cross the line
down_lane3_count = 0  # Counter for vehicles that cross the line

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        # 發送 line 通知
        data['content'] = '串流中斷'
        response = requests.post(url, json=data)
        print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
        cap = cv2.VideoCapture(video_path)     # 有時候串流間隔時間較久會中斷，中斷時重新讀取
        
        continue

    # frame = cv2.resize(frame, (960, 540)) # 704, 480
    
    # Set up the counting line position (adjust as needed)
    up_lane1 = {'p1x':335, 'p1y':230, 'p2x':379, 'p2y':230, }
    up_lane2 = {'p1x':383, 'p1y':230, 'p2x':427, 'p2y':230, }
    up_lane3 = {'p1x':430, 'p1y':230, 'p2x':474, 'p2y':230, }
    up_lane4 = {'p1x':515, 'p1y':230, 'p2x':600, 'p2y':230, }
    down_lane1 = {'p1x':260, 'p1y':280, 'p2x':325, 'p2y':280, }
    down_lane2 = {'p1x':190, 'p1y':280, 'p2x':255, 'p2y':280, }
    down_lane3 = {'p1x':120, 'p1y':180, 'p2x':210, 'p2y':180, }

    # Draw the counting line
    cv2.line(frame, (up_lane1['p1x'], up_lane1['p1y']), (up_lane1['p2x'], up_lane1['p2y']), (0, 0, 255), 1)
    cv2.line(frame, (up_lane2['p1x'], up_lane2['p1y']), (up_lane2['p2x'], up_lane2['p2y']), (255, 0, 0), 1)
    cv2.line(frame, (up_lane3['p1x'], up_lane3['p1y']), (up_lane3['p2x'], up_lane3['p2y']), (0, 0, 255), 1)
    cv2.line(frame, (up_lane4['p1x'], up_lane4['p1y']), (up_lane4['p2x'], up_lane4['p2y']), (255, 0, 0), 1)
    cv2.line(frame, (down_lane1['p1x'], down_lane1['p1y']), (down_lane1['p2x'], down_lane1['p2y']), (255, 0, 0), 1)
    cv2.line(frame, (down_lane2['p1x'], down_lane2['p1y']), (down_lane2['p2x'], down_lane2['p2y']), (0, 0, 255), 1)
    cv2.line(frame, (down_lane3['p1x'], down_lane3['p1y']), (down_lane3['p2x'], down_lane3['p2y']), (255, 0, 0), 1)

    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True)

    # get current time
    current_time = datetime.now()
    current_day_str = current_time.strftime("%Y-%m-%d")

    try:
        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        # Visualize the results on the frame
        frame = results[0].plot()

        current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # Process each tracked object
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box  # Get center coordinates
            track = track_history[track_id]
            track.append((float(x), float(y)))  # Store x, y center point of the track
            if len(track) > 30:  # Keep track history of 30 frames
                track.pop(0)

            # Check if the object crosses the counting line
            if track_id not in counted_ids and y > (up_lane1['p1y'] - 6) and y < (up_lane1['p2y'] + 6):
                if up_lane1['p1x'] < x < up_lane1['p2x']:  # Ensure the vehicle is within the line's x bounds
                    counted_ids.add(track_id)  # Mark vehicle as counted
                    up_lane1_count += 1  # Increment the count

                    # Draw bounding box for the counted vehicle
                    top_left = (int(x - w / 2), int(y - h / 2))
                    bottom_right = (int(x + w / 2), int(y + h / 2))
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 3)  # Green bounding box
                    # cv2.putText(frame, f"ID: {track_id}", (top_left[0], top_left[1] - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    class_name = class_names[class_id]  # 假設 class_names 是一個包含類別名稱的列表
                    # Write to a text file
                    with open(f"{current_day_str}.txt", "a") as file:  # 使用附加模式
                        file.write(f"Lane: up_lane1, Type: {class_name}, Time: {current_time}\n")
                
                elif up_lane2['p1x'] < x < up_lane2['p2x']:  # Ensure the vehicle is within the line's x bounds
                    counted_ids.add(track_id)  # Mark vehicle as counted
                    up_lane2_count += 1  # Increment the count

                    # Draw bounding box for the counted vehicle
                    top_left = (int(x - w / 2), int(y - h / 2))
                    bottom_right = (int(x + w / 2), int(y + h / 2))
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 3)

                    class_name = class_names[class_id]  # 假設 class_names 是一個包含類別名稱的列表
                    # Write to a text file
                    with open(f"{current_day_str}.txt", "a") as file:  # 使用附加模式
                        file.write(f"Lane: up_lane2, Type: {class_name}, Time: {current_time}\n")
                    
                elif up_lane3['p1x'] < x < up_lane3['p2x']:  # Ensure the vehicle is within the line's x bounds
                    counted_ids.add(track_id)  # Mark vehicle as counted
                    up_lane3_count += 1  # Increment the count

                    # Draw bounding box for the counted vehicle
                    top_left = (int(x - w / 2), int(y - h / 2))
                    bottom_right = (int(x + w / 2), int(y + h / 2))
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 3)

                    class_name = class_names[class_id]  # 假設 class_names 是一個包含類別名稱的列表
                    # Write to a text file
                    with open(f"{current_day_str}.txt", "a") as file:  # 使用附加模式
                        file.write(f"Lane: up_lane3, Type: {class_name}, Time: {current_time}\n")
                    
                elif up_lane4['p1x'] < x < up_lane4['p2x']:  # Ensure the vehicle is within the line's x bounds
                    counted_ids.add(track_id)  # Mark vehicle as counted
                    up_lane4_count += 1  # Increment the count

                    # Draw bounding box for the counted vehicle
                    top_left = (int(x - w / 2), int(y - h / 2))
                    bottom_right = (int(x + w / 2), int(y + h / 2))
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 3)

                    class_name = class_names[class_id]  # 假設 class_names 是一個包含類別名稱的列表
                    # Write to a text file
                    with open(f"{current_day_str}.txt", "a") as file:  # 使用附加模式
                        file.write(f"Lane: up_lane4, Type: {class_name}, Time: {current_time}\n")
            
            # Check if the object crosses the counting line
            if track_id not in counted_ids and y < (down_lane1['p1y'] + 6) and y > (down_lane1['p2y'] - 6):
                if down_lane1['p1x'] < x < down_lane1['p2x']:  # Ensure the vehicle is within the line's x bounds
                    counted_ids.add(track_id)  # Mark vehicle as counted
                    down_lane1_count += 1  # Increment the count

                    # Draw bounding box for the counted vehicle
                    top_left = (int(x - w / 2), int(y - h / 2))
                    bottom_right = (int(x + w / 2), int(y + h / 2))
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 3)  # Green bounding box

                    class_name = class_names[class_id]  # 假設 class_names 是一個包含類別名稱的列表
                    # Write to a text file
                    with open(f"{current_day_str}.txt", "a") as file:  # 使用附加模式
                        file.write(f"Lane: down_lane1, Type: {class_name}, Time: {current_time}\n")

                elif down_lane2['p1x'] < x < down_lane2['p2x']:  # Ensure the vehicle is within the line's x bounds
                    counted_ids.add(track_id)  # Mark vehicle as counted
                    down_lane2_count += 1  # Increment the count

                    # Draw bounding box for the counted vehicle
                    top_left = (int(x - w / 2), int(y - h / 2))
                    bottom_right = (int(x + w / 2), int(y + h / 2))
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 3)
                    
                    class_name = class_names[class_id]  # 假設 class_names 是一個包含類別名稱的列表
                    # Write to a text file
                    with open(f"{current_day_str}.txt", "a") as file:  # 使用附加模式
                        file.write(f"Lane: down_lane2, Type: {class_name}, Time: {current_time}\n")
                    
                elif down_lane3['p1x'] < x < down_lane3['p2x']:  # Ensure the vehicle is within the line's x bounds
                    counted_ids.add(track_id)  # Mark vehicle as counted
                    down_lane3_count += 1  # Increment the count

                    # Draw bounding box for the counted vehicle
                    top_left = (int(x - w / 2), int(y - h / 2))
                    bottom_right = (int(x + w / 2), int(y + h / 2))
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 3)
                    
                    class_name = class_names[class_id]  # 假設 class_names 是一個包含類別名稱的列表
                    # Write to a text file
                    with open(f"{current_day_str}.txt", "a") as file:  # 使用附加模式
                        file.write(f"Lane: down_lane1, Type: {class_name}, Time: {current_time}\n")
    
    except:
        None
    finally:
        # Display the vehicle count in the bottom-right corner
        cv2.putText(frame, f"{down_lane3_count} {down_lane2_count} {down_lane1_count} {up_lane1_count} {up_lane2_count} {up_lane3_count} {up_lane4_count}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the annotated frame
        # cv2.imshow("YOLO11 Tracking", frame)
    
    # 檢查目前時間是否為整點
    current_hour = current_time.hour
    if last_hour != current_hour and last_hour != None:
        # 發送整點通知
        data['content'] = f"{current_day_str} {last_hour}~{current_hour}\n\n\
下行車道3： {down_lane3_count}\n\
下行車道2： {down_lane2_count}\n\
下行車道1： {down_lane1_count}\n\
上行車道1： {up_lane1_count}\n\
上行車道2： {up_lane2_count}\n\
上行車道3： {up_lane3_count}\n\
上行車道4： {up_lane4_count}\n\
=============================="
        response = requests.post(url, json=data)

        # 更新記錄的時間
        last_hour = current_hour
        down_lane3_count = 0
        down_lane2_count = 0
        down_lane1_count = 0
        up_lane1_count = 0
        up_lane2_count = 0
        up_lane3_count = 0
        up_lane4_count = 0

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# 發送 line 通知
data['content'] = '程式停止'
response = requests.post(url, json=data)
