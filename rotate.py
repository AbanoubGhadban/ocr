import cv2
import numpy as np

# rearrange the points
# sort points of rectangle in the following order
# top-left, top-right, bottom-right, bottom-left
def rectify(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    tmp = np.argmin(add)
    hnew[0] = h[tmp]
    # delete choosen point to ensure it will not picked again
    add = np.delete(add, tmp)
    h = np.delete(h, tmp, 0)

    tmp = np.argmax(add)
    hnew[2] = h[tmp]
    h = np.delete(h, tmp, 0)    

    diff = np.diff(h,axis = 1)
    tmp = np.argmin(diff)
    hnew[1] = h[tmp]
    hnew[3] = h[0 if tmp == 1 else 1]
    return hnew

# get the points in manual mode
def get_mouse_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x,y])

# get points from user
points = []

# set mode "M" for Manual or any this else for Atomatic
mode = "M"

# read the imgage
image = cv2.imread('doc3.jpg')
orig = image.copy()

if mode == "M":
    cv2.namedWindow('select_points', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("select_points", get_mouse_points)
    
    while(1):
        cv2.imshow("select_points", image)
        key = cv2.waitKey(20) & 0xFF

        # draw circle in each point
        for point in points:
            cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)
        
        # break if 4 points collected
        if(len(points) == 4):
            print(points)
            cv2.destroyAllWindows()
            target = np.array(points)
            break
else:
    # resize image so it can be processed
    # choose optimal dimensions such that important content is not lost
    image = cv2.resize(image, (1500, 800))

    # convert to grayscale and blur to smooth
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # apply Canny Edge Detection
    edged = cv2.Canny(blurred, 0, 50)
    orig_edged = edged.copy()

    # find the contours in the edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # get approximate contour
    for c in contours:
        p = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * p, True)
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 9)
        if len(approx) == 4:
            target = approx
            break

# mapping target points to 800x800 quadrilateral
approx = rectify(target)
cv2.drawContours(image, [approx.astype(int)], -1, (0, 255, 0), 9)
pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])

M = cv2.getPerspectiveTransform(approx,pts2)
dst = cv2.warpPerspective(orig,M,(800,800))

# using thresholding on warped image to get scanned effect
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
ret,th1 = cv2.threshold(dst,127,255,cv2.THRESH_BINARY)

th2 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
ret2,th4 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.namedWindow('photo', cv2.WINDOW_NORMAL)
cv2.imshow("photo", dst)


cv2.waitKey(0)
cv2.destroyAllWindows()