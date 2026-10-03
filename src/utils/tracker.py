import cv2
import mongoengine as db
from ultralytics import YOLO
from src.src.core.machine_learning.computer_vision.dfmd_tracking_algorithm import Tracker

'''
The script is responsible for the collection of data from a video file.
It takes the following inputs:
        1. db_name -> str(), name under which the data will be saved in the database
        2. ref_img -> str(), path to reference image
        3. model -> YOLO initalized model
        4. video_file -> str(), path the video file

It save the metadata collected by the tracker from the video file in the database.
'''

def metadata_collection(db_name, ref_img, model, video_file):
    db.connect(db_name)
    tracker_object = Tracker(ref_img, model, video_file)

    cap = cv2.VideoCapture(video_file)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    frame_count = 1
    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        tracker_object.extract_metadata(im0, frame_count)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # initializing the inputs ------>
    db_name = 'Video21'
    ref_img = 'res/frame_12560.png'
    model = YOLO('models/nano_human_detection_14_Mar_2024.pt')
    video_file = 'test_videos\\dfmd_1_min.mp4'
    
    # running the main function ------>
    metadata_collection(db_name, ref_img, model, video_file)
