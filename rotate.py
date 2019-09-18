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
image = cv2.imread('images/test1.jpg')
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
(tl, tr, br, bl) = approx = rectify(target)
cv2.drawContours(image, [approx.astype(int)], -1, (0, 255, 0), 9)

# compute the width of the new image, which will be the
# maximum distance between bottom-right and bottom-left
# x-coordiates or the top-right and top-left x-coordinates
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

# compute the height of the new image, which will be the
# maximum distance between the top-right and bottom-right
# y-coordinates or the top-left and bottom-left y-coordinates
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

# now that we have the dimensions of the new image, construct
# the set of destination points to obtain a "birds eye view",
# (i.e. top-down view) of the image, again specifying points
# in the top-left, top-right, bottom-right, and bottom-left
# order
dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

# compute the perspective transform matrix and then apply it
M = cv2.getPerspectiveTransform(approx, dst)
warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# using thresholding on warped image to get scanned effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ret,th1 = cv2.threshold(warped,127,255,cv2.THRESH_BINARY)

th2 = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
ret2,th4 = cv2.threshold(warped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.namedWindow('photo', cv2.WINDOW_NORMAL)
cv2.imshow("photo", warped)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow("img", image)


cv2.waitKey(0)
cv2.destroyAllWindows()