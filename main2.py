import cv2, time, csv, random, numpy as np
from ultralytics import YOLO
import cvzone

# Load YOLO
model = YOLO("yolov8n.pt")  # or "yolo11s.pt"
names = model.names

# Floor videos (replace with your actual 4 videos)
floor_paths = ["lift1.mp4", "lift2.mp4", "lift3.mp4", "lift4.mp4"]
caps_floor = [cv2.VideoCapture(p) for p in floor_paths]

# Lift settings
MAX_CAPACITY = 8
TRAVEL_TIME_PER_FLOOR = 5   # seconds
STOP_TIME = 8               # how long the lift stops at each floor
PROCESS_EVERY_N_FRAMES = 4  # skip frames for speed
FPS_SIM = 5                 # simulation speed (slower = smaller number)

inside_count = 0
waiting_counts = [0]*len(caps_floor)
frame_idx = 0
current_floor = 0
direction = 1  # 1 = down, -1 = up

# CSV log setup
with open("lift_dashboard_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time","frame","floor","inside_before","waiting_before","boarded","inside_after","waiting_after","eta_next"])

stop_timer = 0  # how long lift has been stopped

while True:
    frame_idx += 1
    frames = []
    for i, cap in enumerate(caps_floor):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video
            ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 240))  # smaller for grid
        frames.append(frame)

    # detect waiting people at all floors
    for i, frame in enumerate(frames):
        results = model(frame)
        waiting = 0
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            cls = results[0].boxes.cls.int().cpu().tolist()
            for box, c in zip(boxes, cls):
                if names[c] == "person":
                    waiting += 1
                    x1,y1,x2,y2 = box
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        waiting_counts[i] = waiting
        cvzone.putTextRect(frame, f"F{i+1}: {waiting}", (5,20), 1, 1)

    # simulate lift stop
    if stop_timer > 0:
        stop_timer -= 1
    else:
        # lift arrives at a new floor
        inside_before = inside_count
        waiting_before = waiting_counts[current_floor]
        space_left = max(0, MAX_CAPACITY - inside_count)
        boarded = random.randint(0, min(space_left, waiting_before))
        inside_count += boarded
        waiting_counts[current_floor] -= boarded
        eta_next = TRAVEL_TIME_PER_FLOOR * (len(caps_floor)-current_floor-1)
        with open("lift_dashboard_log.csv","a",newline="") as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), frame_idx, current_floor+1, inside_before, waiting_before, boarded, inside_count, waiting_counts[current_floor], eta_next])
        stop_timer = STOP_TIME * FPS_SIM
        # move floor pointer
        current_floor += direction
        if current_floor == len(caps_floor)-1 or current_floor == 0:
            direction *= -1

    # make grid of videos (2x2)
    top = cv2.hconcat([frames[0], frames[1]])
    bottom = cv2.hconcat([frames[2], frames[3]])
    grid = cv2.vconcat([top, bottom])

    # status panel
    panel = np.zeros((480, 320, 3), dtype=np.uint8)
    cvzone.putTextRect(panel, f"Lift Status", (10,40), 2, 2)
    cvzone.putTextRect(panel, f"Current Floor: {current_floor+1}", (10,100), 1,1)
    cvzone.putTextRect(panel, f"Inside: {inside_count}/{MAX_CAPACITY}", (10,140), 1,1)
    waits = " | ".join([f"F{i+1}:{waiting_counts[i]}" for i in range(len(waiting_counts))])
    cvzone.putTextRect(panel, waits, (10,200), 1,1)
    cvzone.putTextRect(panel, f"ETA next floor: {TRAVEL_TIME_PER_FLOOR}s", (10,240), 1,1)

    # combine grid and panel
    combined = cv2.hconcat([grid, panel])

    cv2.imshow("Lift Multi-Floor Simulation", combined)
    if cv2.waitKey(int(1000/FPS_SIM)) & 0xFF == ord("q"):
        break

for cap in caps_floor: cap.release()
cv2.destroyAllWindows()