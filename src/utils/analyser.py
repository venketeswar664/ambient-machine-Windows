import cv2
import mongoengine as db
from ultralytics import YOLO
from src.src.core.machine_learning.computer_vision.dfmd_tracking_algorithm import Tracker

'''
The script is responsible for analysing the data collected from a video file and returning a visual representative
output of the video according to the usecase logic determining weather frisking is done or not.

It takes the following inputs:
        1. db_name -> str(), name under which the data will be saved in the database
        2. ref_img -> str(), path to reference image
        3. model -> YOLO initalized model
        4. video_file -> str(), path the video file
        5. output_video_name -> str(), name of the output video file in '.avi' format.

It returns a output video file in '.avi' format
'''

def video_analser(db_name, ref_img, model, video_file, output_video_name):
    db.connect(db_name)
    tracker_object = Tracker(ref_img, model, video_file)

    tracker_object.process_metadata(output_video=output_video_name)

if __name__ == '__main__':
    # initializing the inputs ------>
    db_name = 'Video1113'
    ref_img = 'res/frame_12560.png'
    model = YOLO('models/nano_human_detection_14_Mar_2024.pt')
    video_file = 'test_videos\\ch01_00000000000000013.mp4'
    output_video_name = 'result.avi'

    # running the main function ------>
    video_analser(db_name, ref_img, model, video_file, output_video_name)
