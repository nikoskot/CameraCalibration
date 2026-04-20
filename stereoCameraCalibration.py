import cv2 as cv
import os
import numpy as np
import rerun as rr
import configargparse
from datetime import datetime
import yaml
import json
import time
import tqdm
from pathlib import Path
import monocularCameraCalibration as monoCalib


def getParser():
    parser = configargparse.ArgParser(default_config_files=[".\stereoCalibrationConfig.yaml"])
    parser.add("--configFile", is_config_file=True, help='config file path')
    # parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add("--imagesFolder", type=lambda p: Path(p).resolve(), default=".\calibrationImages")
    parser.add("--liveCapture", action="store_true")
    # parser.add("--imagesGroup", type=str, choices=["left", "right"], default="all")
    parser.add("--patternRowCorners", type=int, default=9)
    parser.add("--patternColumnCorners", type=int, default=6)
    parser.add("--patternGridSize", type=int, default=1)
    parser.add("--dontRefineCorners", action="store_true")
    parser.add("--resultsSavePath", type=lambda p: Path(p).resolve(), default=".\stereoCalibrationResults")
    return parser


def saveArgsToYaml(args, filename):
    # Convert to dict and dump to YAML file
    with open(filename, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def captureCalibrationImagesFromTwoCameras():
    cameraWidth = 1920
    cameraHeight = 1080
    cameraFps = 60
    
    cap0 = cv.VideoCapture(1)
    if not cap0.isOpened():
        print("Cannot open camera 0.")
        return
    else: 
        print("Camera 0 opened.")
    cap0.set(cv.CAP_PROP_FRAME_WIDTH, cameraWidth)   # width in pixels
    cap0.set(cv.CAP_PROP_FRAME_HEIGHT, cameraHeight)   # height in pixels
    cap0.set(cv.CAP_PROP_FPS, cameraFps)  # frames per second
    
    cap1 = cv.VideoCapture(0)
    if not cap1.isOpened():
        print("Cannot open camera 1.")
        return
    else: 
        print("Camera 1 opened.")
    cap1.set(cv.CAP_PROP_FRAME_WIDTH, cameraWidth)   # width in pixels
    cap1.set(cv.CAP_PROP_FRAME_HEIGHT, cameraHeight)   # height in pixels
    cap1.set(cv.CAP_PROP_FPS, cameraFps)

    saveInterval = 3.0  # seconds between saves
    saving = False
    lastSaveTime = 0

    frameCount = 0  # count saved frames
    leftImages = []
    rightImages = []
    info0 = "Camera 0 (Left). 's' tp start saving. 'q' to stop. 'f' to stop and switch left to right images"
    info1 = "Camera 1 (Right). 's' tp start saving. 'q' to stop. 'f' to stop and switch left to right images"
    
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not (ret0 and ret1):
            print("Can't receive frame from at least one camera (stream end?). Exiting ...")
            break

        currentTime = time.time()
        frame0Copy = cv.resize(cv.flip(frame0.copy(), 1), (640, 360))
        frame1Copy = cv.resize(cv.flip(frame1.copy(), 1), (640, 360))

        # If saving mode started, check time and save frames every "save_interval" seconds
        if saving:
            elapsed = currentTime - lastSaveTime

            # Calculate countdown (seconds remaining to next save)
            countdown = max(0, saveInterval - elapsed)
            countdownText = f"Next capture in: {countdown:.1f}s"

            # Put countdown text on frame
            cv.putText(frame0Copy, countdownText, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv.LINE_AA)

            if elapsed >= saveInterval:
                # Save frame
                leftImages.append(frame0)
                rightImages.append(frame1)
                frameCount += 1
                lastSaveTime = currentTime

        else:
            # Show instruction
            cv.putText(frame0Copy, f"Press 's' to start saving every {saveInterval} seconds", (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow(info0, frame0Copy)
        cv.imshow(info1, frame1Copy)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on 'q'
            break
        if key == ord('f'):  # Quit and switch left and right images on 'f'. In case video capture has wrong camera order
            leftImagesCopy = leftImages.copy()
            leftImages = rightImages.copy()
            rightImages = leftImagesCopy
            break
        elif key == ord('s'):  # Start saving on 's'
            if not saving:
                cv.destroyAllWindows()
                saving = True
                lastSaveTime = currentTime

    cap0.release()
    cap1.release()
    cv.destroyAllWindows()
    print(f"Captured {len(leftImages)} left and {len(rightImages)} right images.") 
    return leftImages, rightImages


def visualizeSetup(R, T, K1, K2, height=1080, width=1920):
    # rr.init("stereo_calibration", spawn=True)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Left camera (world origin)
    rr.log("world/left_cam", rr.Transform3D(mat3x3=R, translation=T.flatten()), static=True)
    # rr.log("left_cam/frame", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/left_cam/axes", rr.Arrows3D(origins=np.zeros((3, 3)), vectors=np.eye(3), colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]), static=True)
    rr.log("world/left_cam/image", rr.Pinhole(image_from_camera=K1, resolution=[width, height]), static=True)
    # Right camera
    rr.log("world/right_cam", rr.Transform3D(mat3x3=np.eye(3), translation=[0,0,0]), static=True)
    rr.log("world/right_cam/axes", rr.Arrows3D(origins=np.zeros((3, 3)), vectors=np.eye(3), colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]), static=True)
    rr.log("world/right_cam/image", rr.Pinhole(image_from_camera=K2, resolution=[width, height]), static=True)


def stereoCameraCalibration(leftImages, rightImages, nCornersPerRow=9, nCornersPerColumn=6, patternGridSize=1, refineCorners=True, savePath=None):
    worldCoordsSingle = np.zeros((nCornersPerRow*nCornersPerColumn, 3), np.float32)
    worldCoordsSingle[:, :2] = np.mgrid[0:nCornersPerRow, 0:nCornersPerColumn].T.reshape(-1, 2) * patternGridSize
    leftStereoImageCoords = []
    rightStereoImageCoords = [] 
    stereoWorldCoords = []
    leftMonoImageCoords = []
    rightMonoImageCoords = []
    leftMonoWorldCoords = []
    rightMonoWorldCoords = []
    h, w, _ = leftImages[0].shape
    
    print("Detecting pattern cornenrs.")
    for i in tqdm.tqdm(range(len(leftImages))):
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret0, corners0 = cv.findChessboardCorners(leftImages[i], (nCornersPerRow, nCornersPerColumn))
        ret1, corners1 = cv.findChessboardCorners(rightImages[i], (nCornersPerRow, nCornersPerColumn))
        
        # If there is a pattern detected in the left image, use it for the monoculat calibration of the left camera for sure
        if ret0:
            if refineCorners:
                corners0 = cv.cornerSubPix(cv.cvtColor(leftImages[i], cv.COLOR_BGR2GRAY), corners0, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            if savePath:
                imgCopy = leftImages[i].copy()
                cv.drawChessboardCorners(imgCopy, (nCornersPerRow, nCornersPerColumn), corners0, ret0)
                cv.imwrite(Path.joinpath(savePath, f"annotatedLeftMono{i}.png"), imgCopy)
            leftMonoImageCoords.append(corners0.reshape(-1, 2))
            leftMonoWorldCoords.append(worldCoordsSingle)
        
        # If there is a pattern detected in the right image, use it for the monoculat calibration of the right camera for sure
        if ret1:
            if refineCorners:
                corners1 = cv.cornerSubPix(cv.cvtColor(rightImages[i], cv.COLOR_BGR2GRAY), corners1, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            if savePath:
                imgCopy = rightImages[i].copy()
                cv.drawChessboardCorners(imgCopy, (nCornersPerRow, nCornersPerColumn), corners1, ret1)
                cv.imwrite(Path.joinpath(savePath, f"annotatedRightMono{i}.png"), imgCopy)
            rightMonoImageCoords.append(corners1.reshape(-1, 2))
            rightMonoWorldCoords.append(worldCoordsSingle)
        
        # If either of the images does not have the pattern visible, do not use it for stereo calibration    
        if not ret0:
            print(f"No chessboard corners found in left image {i}")
            cv.imshow(f"No corners {i}", leftImages[i])
            cv.waitKey(1000)
            cv.destroyAllWindows()
            continue
        if not ret1:
            print(f"No chessboard corners found in right image {i}")
            cv.imshow(f"No corners {i}", rightImages[i])
            cv.waitKey(1000)
            cv.destroyAllWindows()
            continue
            
        if savePath:
            imgCopy = leftImages[i].copy()
            cv.drawChessboardCorners(imgCopy, (nCornersPerRow, nCornersPerColumn), corners0, ret0)
            cv.imwrite(Path.joinpath(savePath, f"annotatedLeftStereo{i}.png"), imgCopy)
            imgCopy = rightImages[i].copy()
            cv.drawChessboardCorners(imgCopy, (nCornersPerRow, nCornersPerColumn), corners1, ret1)
            cv.imwrite(Path.joinpath(savePath, f"annotatedRightStereo{i}.png"), imgCopy)

        leftStereoImageCoords.append(corners0.reshape(-1, 2))
        stereoWorldCoords.append(worldCoordsSingle)
        rightStereoImageCoords.append(corners1.reshape(-1, 2))

    print("Calibrating.")
    start = time.time()
    leftRmse, leftCameraMatrix, leftDistortionCoeffs, leftRotationVecs, leftTranslationVecs = monoCalib.opencvSingleCameraCalibration(leftImages, leftMonoWorldCoords, leftMonoImageCoords)
    rightRmse, rightCameraMatrix, rightDistortionCoeffs, rightRotationVecs, rightTranslationVecs = monoCalib.opencvSingleCameraCalibration(rightImages, rightMonoWorldCoords, rightMonoImageCoords)

    print(f"Left monocular calibration RMSE (pixels): \n{leftRmse}")
    print(f"Left camera matrix: \n{leftCameraMatrix}")
    print(f"Left camera distortion coefficients: \n{leftDistortionCoeffs}")
    
    print(f"Right monocular calibration RMSE (pixels): \n{rightRmse}")
    print(f"Right camera matrix: \n{rightCameraMatrix}")
    print(f"Right camera distortion coefficients: \n{rightDistortionCoeffs}")

    stereoRmse, leftCameraMatrix, leftDistortionCoeffs, rightCameraMatrix, rightDistortionCoeffs, R, T, E, F = cv.stereoCalibrate(
        stereoWorldCoords, 
        leftStereoImageCoords, 
        rightStereoImageCoords,
        leftCameraMatrix, 
        leftDistortionCoeffs, 
        rightCameraMatrix, 
        rightDistortionCoeffs,
        (w, h),
        flags=cv.CALIB_FIX_INTRINSIC
    )
    
    print(f"Stereo setup calibration took {time.time() - start}.")

    print(f"Stereo calibration RMSE: \n{stereoRmse}")
    print(f"Rotation between cameras: \n{R}")
    print(f"Translation between cameras: \n{T}")
    print(f"Essential Matrix: \n{E}")
    print(f"Fundamental Matrix: \n{F}")
    
    return stereoRmse, leftCameraMatrix, leftDistortionCoeffs, rightCameraMatrix, rightDistortionCoeffs, R, T, E, F


def saveCalibrationParams(folderPath, stereoRmse, leftCameraMatrix, leftDistortionCoeffs, rightCameraMatrix, rightDistortionCoeffs, R, T, E, F):
    try:
        params = {
            "leftCameraMatrix": leftCameraMatrix.tolist(),
            "leftDistortionCoeffs": leftDistortionCoeffs.tolist(),
            "rightCameraMatrix": rightCameraMatrix.tolist(),
            "rightDistortionCoeffs": rightDistortionCoeffs.tolist(),
            "rmse": stereoRmse,
            "R": R.tolist(),
            "T": T.tolist(),
            "E": E.tolist(),
            "F": F.tolist(),
        }
        
        with open(Path.joinpath(folderPath, "stereoCalib.json"), "w") as f:
            json.dump(params, f, indent=2)
            
    except Exception as e:
        print(f"Could not save stereo calibration results in file {folderPath}\stereoCalib.json. \n Exception {e}")
      

def loadCalibrationParams(folderPath):
    try:
        with open(Path.joinpath(folderPath, "stereoCalib.json"), "r") as f:
            data = json.load(f)
            
        params = {
            "leftCameraMatrix": np.array(data["leftCameraMatrix"], dtype=np.float64),
            "leftDistortionCoeffs": np.array(data["leftDistortionCoeffs"], dtype=np.float64),
            "rightCameraMatrix": np.array(data["rightCameraMatrix"], dtype=np.float64),
            "rightDistortionCoeffs": np.array(data["rightDistortionCoeffs"], dtype=np.float64),
            "rmse": data["rmse"],
            "R": np.array(data["R"], dtype=np.float64),
            "T": np.array(data["T"], dtype=np.float64),
            "E": np.array(data["E"], dtype=np.float64),
            "F": np.array(data["F"], dtype=np.float64),
        }
        
        return params
    
    except Exception as e:
        print(f"Could not load calibration results from file {folderPath}\calib.json. \n Exception {e}")

        
def main():
    parser = getParser()
    args = parser.parse_args()

    # Create necessary folders/paths
    print("---Creating path for calibration results.---")
    args.resultsSavePath = Path.joinpath(args.resultsSavePath, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.resultsSavePath, exist_ok=True)
    os.makedirs(Path.joinpath(args.resultsSavePath, "annotatedImages"), exist_ok=True)
    # Save arguments use to file
    saveArgsToYaml(args, Path.joinpath(args.resultsSavePath, "config.yaml"))

    print(f"---Starting stereo camera calibration with: \n {vars(args)}---")

    if args.liveCapture:
        print("---Capturing calibration images from two cameras.---")
        leftImages, rightImages = captureCalibrationImagesFromTwoCameras()
    else:
        print(f"---Loading images from folder {args.imagesFolder}.---")
        leftImages, rightImages = monoCalib.loadImages(group="all", folderName=args.imagesFolder)
    if len(leftImages) == 0 or len(rightImages) == 0:
        print("At least one set of images is empty. Quitting.")
        return
    if len(leftImages) != len(rightImages):
        print("Left/Right set of images are not the same number. Quitting.")
        return
    
    # showImagesInGrid(leftImages)
    # showImagesInGrid(rightImages)

    stereoRmse, leftCameraMatrix, leftDistortionCoeffs, rightCameraMatrix, rightDistortionCoeffs, R, T, E, F = stereoCameraCalibration(
        leftImages=leftImages, 
        rightImages=rightImages, 
        nCornersPerRow=args.patternRowCorners, 
        nCornersPerColumn=args.patternColumnCorners,
        patternGridSize=args.patternGridSize, 
        refineCorners=(not args.dontRefineCorners), 
        savePath=Path.joinpath(args.resultsSavePath, "annotatedImages")
        )

    print(f"---Saving calibration results to folder {args.resultsSavePath}---")
    saveCalibrationParams(args.resultsSavePath, stereoRmse, leftCameraMatrix, leftDistortionCoeffs, rightCameraMatrix, rightDistortionCoeffs, R, T, E, F)
    
    print(f"Loading calibration results from folder {args.resultsSavePath}")
    calibrationParams = loadCalibrationParams(args.resultsSavePath)
    print(f"Loaded calibration parameters: \n {calibrationParams}")
    
    rr.init("stereo_calibration", spawn=True)
    visualizeSetup(R, T, leftCameraMatrix, rightCameraMatrix, height=leftImages[0].shape[0], width=leftImages[0].shape[1])
    

if __name__ == '__main__':
    main()
