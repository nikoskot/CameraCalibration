import cv2 as cv
import numpy as np
from monocularCameraCalibration import monocularCameraCalibration, loadImages

def undistrortImages(images, K, distortionCoeffs):
    fx, fy = K[0,0], K[1,1]
    s = K[0,1]
    cx, cy = K[0,2], K[1,2]
    k1, k2, p1, p2 = distortionCoeffs.flatten()[:4]

    undistortedImages = []
    for img in images:
        h, w = img.shape[:2]
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        x = ((u - cx) - s*(v - cy)/fy) / fx
        y = (v - cy) / fy

        r_2 = x*x + y*y
        L = 1 + k1*r_2 + k2*r_2*r_2

        Dx = 2*p1*x*y + p2*(r_2 + 2*x*x)
        Dy = p1*(r_2 + 2*y*y) + 2*p2*x*y

        x_d = x*L + Dx
        y_d = y*L + Dy

        u_d = fx*x_d + s*y_d + cx
        v_d = fy*y_d + cy

        map1 = u_d.astype(np.float32)
        map2 = v_d.astype(np.float32)
        undistorted = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
        undistortedImages.append(undistorted)
    
    return undistortedImages

def undistrortImagesOpenCV(images, K, distortionCoeffs):

    undistortedImages = []
    for img in images:
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(K, distortionCoeffs.flatten(), (w,h), 1, (w,h))

        # undistort
        dst = cv.undistort(img, K, distortionCoeffs.flatten(), None, newcameramtx)
        
        # crop the image
        x, y, w, h = roi
        undistorted = dst[y:y+h, x:x+w]
        undistortedImages.append(undistorted)
    
    return undistortedImages

if __name__ == '__main__':

    images = loadImages("left")

    K, Rs, Ts, distortionCoeffs = monocularCameraCalibration(images, 9, 6, True)

    print(f"Camera Matrix: \n{K}")
    print(f"Distortion coefficients: \n{distortionCoeffs}")

    manuallyUndistortedImages = undistrortImages(images, K, distortionCoeffs)

    opencvUndistortedImages = undistrortImagesOpenCV(images, K, distortionCoeffs)

    for original, manual, opencv in zip(images, manuallyUndistortedImages, opencvUndistortedImages):
        cv.imshow("Original", original)
        cv.imshow("Manually Undistorted", manual)
        cv.imshow("OpenCV Undistorted", opencv)
        cv.waitKey(0)
        cv.destroyAllWindows()
