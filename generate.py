import glob
import shutil
import cv2
import os


def frameExtractor(videopath, frames_path, count, prefix, applyMod):
    if applyMod:
        count = int(count % 17)
    saveLocation = frames_path + "/" + str(count)
    if not os.path.exists(saveLocation):
        os.mkdir(saveLocation)
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    start = int(video_length * 0.5)
    number = 1
    for i in range(start - 2, start + 2, 1):
        cap.set(1, i)
        ret, img_arr = cap.read()
        img_arr = cv2.resize(img_arr, (200, 200))
        # img_arr = (cv2.flip(img_arr, 1) + img_arr)
        # img_arr = cv2.multiply(img_arr, 0.5)
        cv2.imwrite(saveLocation + "/" + prefix + "%#05d.png" % number, img_arr)
        number += 1
        # # For generating Flip images
        # cv2.imwrite(saveLocation + "/" + prefix + "%#05d.png" % number, cv2.flip(img_arr, 1))
        # number += 1


def generateTrainingData(inputPathName, multiple=False):
    videos = glob.glob(os.path.join(inputPathName, "*.mp4"))
    frameFolderName = "TrainingData_" + inputPathName
    shutil.rmtree(frameFolderName, ignore_errors=True)
    frames_path = os.path.join(frameFolderName)
    if not os.path.exists(frameFolderName):
        os.makedirs(frameFolderName)
    print("Extracting Frames of " + inputPathName)
    frames = range(17)
    for i, video in enumerate(videos):
        print(str(video))
        frameExtractor(video, frames_path, frames[i % 17], str(video.split("\\")[1].split(".mp4")[0]),
                       multiple)


generateTrainingData("test-prof")
generateTrainingData("traindata")