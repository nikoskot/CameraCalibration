import cv2 as cv
import os
import numpy as np
from pathlib import Path
import plotly.graph_objs as pgo
import plotly.offline as pyo
import tqdm
import time
import configargparse
import yaml
import json
from datetime import datetime


def getParser():
    parser = configargparse.ArgParser(default_config_files=[".\monocularCalibrationConfig.yaml"])
    parser.add("--configFile", is_config_file=True, help='config file path')
    # parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add("--imagesFolder", type=lambda p: Path(p).resolve(), default=".\calibrationImages")
    parser.add("--liveCapture", action="store_true")
    parser.add("--imagesGroup", type=str, choices=["left", "right"], default="left")
    parser.add("--patternRowCorners", type=int, default=9)
    parser.add("--patternColumnCorners", type=int, default=6)
    parser.add("--dontRefineCorners", action="store_true")
    parser.add("--resultsSavePath", type=lambda p: Path(p).resolve(), default=".\monocularCalibrationResults")
    return parser


def saveArgsToYaml(args, filename):
    # Convert to dict and dump to YAML file
    with open(filename, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def loadImages(group="all", folderName=None):
    """
    Load the images of the calibration pattern. 
    """
    if not Path.exists(folderName):
        print(f"The provided folder {folderName} does not exist.")
        return []
    
    imageFiles = os.listdir(folderName)
    if len(imageFiles) == 0:
        print(f"The provided folder {folderName} exists but is empty.")
        return []
    
    if group == "left" or group == "right":
        images = []
        for imgFile in imageFiles:
            if imgFile.startswith(group):
                imgPath = os.path.join(folderName, imgFile)
                images.append(cv.imread(imgPath))
                
        print(f"Loaded {len(images)} {group} images.")
        return images
    
    elif group == "all":
        leftImages, rightImages = [], []
        for imgFile in imageFiles:
            if imgFile.startswith("left"):
                imgPath = os.path.join(folderName, imgFile)
                leftImages.append(cv.imread(imgPath)) 
                
            if imgFile.startswith("right"):
                imgPath = os.path.join(folderName, imgFile)
                rightImages.append(cv.imread(imgPath))
        print(f"Loaded {len(leftImages)} left and {len(rightImages)} right images.")  
        return leftImages, rightImages


def captureCalibrationImagesFromSingleCamera():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)   # width in pixels
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)   # height in pixels

    saveInterval = 3.0  # seconds between saves
    saving = False
    lastSaveTime = 0

    frameCount = 0  # count saved frames
    images = []
    info = "Waiting to start capturing images."
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        currentTime = time.time()
        frameCopy = cv.flip(frame.copy(), 1)

        # If saving mode started, check time and save frames every 2 seconds
        if saving:
            elapsed = currentTime - lastSaveTime

            # Calculate countdown (seconds remaining to next save)
            countdown = max(0, saveInterval - elapsed)
            countdownText = f"Next capture in: {countdown:.1f}s"

            # Put countdown text on frame
            cv.putText(frameCopy, countdownText, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv.LINE_AA)

            if elapsed >= saveInterval:
                # Save frame
                images.append(frame)
                frameCount += 1
                lastSaveTime = currentTime

        else:
            # Show instruction
            cv.putText(frameCopy, f"Press 's' to start saving every {saveInterval} seconds", (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        cv.imshow(info, frameCopy)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on 'q'
            break
        elif key == ord('s'):  # Start saving on 's'
            if not saving:
                cv.destroyAllWindows()
                saving = True
                lastSaveTime = currentTime
                info = "Capturing images...Press q to stop"

    cap.release()
    cv.destroyAllWindows()
    return images
    

def showImagesInGrid(images):

    rows = np.sqrt(len(images)).astype(int)
    cols = len(images) // rows + (len(images) % rows > 0)

    remaining = rows * cols - len(images)

    gridRows = []
    for r in range(rows):
        rowImgs = images[r*cols:(r+1)*cols]
        row = np.hstack(rowImgs)
        if r == rows - 1 and remaining > 0:
            blankImg = np.zeros_like(images[0])
            for _ in range(remaining):
                row = np.hstack((row, blankImg))
        gridRows.append(row)
    grid = np.vstack(gridRows)

    cv.namedWindow("Grid", cv.WINDOW_NORMAL)
    cv.imshow("Grid", grid)
    cv.waitKey(0)
    cv.destroyAllWindows()

def opencvSingleCameraCalibration(images, worldCoords, imageCoords):
    '''
    Calibrate a monocular camera using the OpenCV implementation.
    '''
    h, w, _ = images[0].shape
    reprojectionRMSE, cameraMatrix, distortionCoeffs, rotationVecs, translationVecs = cv.calibrateCamera(worldCoords, imageCoords, (w, h), None, None)
    
    return reprojectionRMSE, cameraMatrix, distortionCoeffs, rotationVecs, translationVecs 


def calculateReprojectionErrorScatterPlot(worldCoords, imageCoords, cameraMatrix, rotationVecs, translationVecs, distortionCoeffs=np.array([])):
    errorsX = []
    errorsY = []
    imgIds = []
    ptsIds = []
    
    for id, (wrld, img, r, t) in enumerate(zip(worldCoords, imageCoords, rotationVecs, translationVecs)):
        projected, _ = cv.projectPoints(wrld, r, t, cameraMatrix, distortionCoeffs)
        projected = projected.reshape(-1, 2)

        errorsX.extend(img.reshape(-1, 2)[:, 0] - projected[:, 0])
        errorsY.extend(img.reshape(-1, 2)[:, 1] - projected[:, 1])
        imgIds.extend([id] * len(img))
        ptsIds.extend(list(range(len(img))))
    
    scatter  = pgo.Scatter(x=errorsX,
                           y=errorsY,
                           mode='markers',
                           marker=dict(size=10, color='royalblue'),
                           text = imgIds,
                           hoverinfo='text'
                           )
    layout = pgo.Layout(
        title='XY Reprojection Error',
        xaxis=dict(
            title='Error X',
            scaleanchor='y',  # Fix aspect ratio 1:1 by linking x axis scale to y axis scale
            scaleratio=1
            ),
        yaxis=dict(title='Error Y')
        )
    fig = pgo.Figure(data=[scatter], layout=layout)
    pyo.plot(fig)


def monocularCameraCalibration(images, nCornersPerRow=9, nCornersPerColumn=6, refineCorners=True, savePath=None):
    """
    This function is used by external code in order to calibrate a single camera using as input the images of the calibration pattern.
    """
    worldCoordsSingle = np.zeros((nCornersPerRow*nCornersPerColumn, 3), np.float32)
    worldCoordsSingle[:, :2] = np.mgrid[0:nCornersPerRow, 0:nCornersPerColumn].T.reshape(-1, 2)
    imageCoords = [] 
    worldCoords = []

    print("Detecting pattern cornenrs.")
    for i, img in tqdm.tqdm(enumerate(images), total=len(images)):
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(img, (nCornersPerRow, nCornersPerColumn))
        if not ret:
            print(f"No chessboard corners found in image {i}")
            cv.imshow(f"No corners {i}", img)
            cv.waitKey(1000)
            cv.destroyAllWindows()
            continue
        
        if refineCorners:
            corners = cv.cornerSubPix(cv.cvtColor(img, cv.COLOR_BGR2GRAY), corners, (11,11), (-1,-1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        
        if savePath:
            imgCopy = img.copy()
            cv.drawChessboardCorners(imgCopy, (nCornersPerRow, nCornersPerColumn), corners, ret)
            cv.imwrite(Path.joinpath(savePath, f"annotated{i}.png"), imgCopy)

        imageCoords.append(corners.reshape(-1, 2))
        worldCoords.append(worldCoordsSingle)

    print("Calibrating.")
    start = time.time()
    reprojectionRMSE, cameraMatrix, distortionCoeffs, rotationVecs, translationVecs = opencvSingleCameraCalibration(images, worldCoords, imageCoords)
    print(f"Camera calibration took {time.time() - start}.")

    print(f"Monocular calibration RMSE (pixels): \n{reprojectionRMSE}")
    print(f"Camera matrix: \n{cameraMatrix}")
    print(f"Camera distortion coefficients: \n{distortionCoeffs}")
    # print(f"Rotation vectors: \n{rotationVecs}")
    # print(f"Translation vectors: \n{translationVecs}")
    
    calculateReprojectionErrorScatterPlot(worldCoords, imageCoords, cameraMatrix, rotationVecs, translationVecs, distortionCoeffs)
    
    return reprojectionRMSE, cameraMatrix, distortionCoeffs, rotationVecs, translationVecs

def saveCalibrationParams(folderPath, rmse, cameraMatrix, distortionCoeffs):
    try:
        params = {
            "cameraMatrix": cameraMatrix.tolist(),
            "distortionCoeffs": distortionCoeffs.tolist(),
            "rmse": rmse,
        }
        
        with open(Path.joinpath(folderPath, "calib.json"), "w") as f:
            json.dump(params, f, indent=2)
            
    except Exception as e:
        print(f"Could not save calibration results in file {folderPath}\calib.json. \n Exception {e}")


def loadCalibrationParams(folderPath):
    try:
        with open(Path.joinpath(folderPath, "calib.json"), "r") as f:
            data = json.load(f)

        params = {
            "cameraMatrix" : np.array(data["cameraMatrix"], dtype=np.float64),
            "distortionCoeffs" : np.array(data["distortionCoeffs"], dtype=np.float64),
            "rmse" : data["rmse"],
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

    print(f"---Starting monocular camera calibration with: \n {vars(args)}---")
    
    images = []
    if args.liveCapture:
        print("---Capturing images from camera.---")
        images = captureCalibrationImagesFromSingleCamera()
    else:
        print(f"---Loading images from folder {args.imagesFolder}.---")
        images = loadImages(group=args.imagesGroup, folderName=args.imagesFolder)
    if len(images) == 0:
        print("No images to use. Quitting.")
        return

    # showImagesInGrid(images)

    reprojectionRMSE, cameraMatrix, distortionCoeffs, rotationVecs, translationVecs = monocularCameraCalibration(images=images, nCornersPerRow=args.patternRowCorners, nCornersPerColumn=args.patternColumnCorners, refineCorners=(not args.dontRefineCorners), savePath=Path.joinpath(args.resultsSavePath, "annotatedImages"))
    
    print(f"---Saving calibration results to folder {args.resultsSavePath}---")
    saveCalibrationParams(args.resultsSavePath, reprojectionRMSE, cameraMatrix, distortionCoeffs)
    
    print(f"Loading calibration results from folder {args.resultsSavePath}")
    calibrationParams = loadCalibrationParams(args.resultsSavePath)
    print(f"Loaded calibration parameters: \n {calibrationParams}")
    
if __name__ == "__main__":
    main()