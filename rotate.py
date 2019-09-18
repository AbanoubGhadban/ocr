import cv2
import numpy as np
import pytesseract
import sys

def getDistance(p1, p2):
    return np.sqrt(((p1[0] - p2[0])**2) + ((p1[1] - p2[1])**2))

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

isClicked = False
# get the points
def get_mouse_points(event, x, y, flags, param):
    global points, isClicked
    if event == cv2.EVENT_LBUTTONDOWN:
        isClicked = True
        points.append([x,y])
    elif isClicked and event == cv2.EVENT_MOUSEMOVE:
        points[-1] = [x, y]
    elif isClicked and event == cv2.EVENT_LBUTTONUP:
        points[-1] = [x, y]
        isClicked = False

# get points from user
points = []

# read the imgage
image = cv2.imread(sys.argv[1])
originalHeight, originalWidth, _ = image.shape
thick = int((originalHeight + originalWidth)/300)

cv2.namedWindow('select_points', cv2.WINDOW_NORMAL)
cv2.resizeWindow('select_points', 800, 600)
cv2.setMouseCallback("select_points", get_mouse_points)

while(1):
    tmpImg = image.copy()
    key = cv2.waitKey(20) & 0xFF
    if (key == ord('c')):
        points = []

    # draw circle in each point
    for point in points:
        cv2.circle(tmpImg, tuple(point), thick, (0, 0, 255), -1)
        if (len(points) == 4):
            cv2.drawContours(tmpImg, [np.array(points)], -1, (0, 255, 0), thick)
        elif (len(points) > 1):
            for i in range(1, len(points)):
                cv2.line(tmpImg, tuple(points[i-1]), tuple(points[i]), (0, 255, 0), thick)
    
    cv2.imshow("select_points", tmpImg)
    # break if 4 points collected
    if(len(points) == 4 and not isClicked):
        print(points)
        cv2.destroyAllWindows()
        target = np.array(points)
        break

# mapping target points to 800x800 quadrilateral
(tl, tr, br, bl) = approx = rectify(target)
cv2.drawContours(image, [approx.astype(int)], -1, (0, 255, 0), 9)

# compute the width of the new image, which will be the
# maximum distance between bottom-right and bottom-left
# x-coordiates or the top-right and top-left x-coordinates
widthA = getDistance(br, bl)
widthB = getDistance(tr, tl)
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
ret2,th4 = cv2.threshold(warped,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

text = pytesseract.image_to_string(th4)
print(text)

cv2.namedWindow('photo', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 800, 600)
cv2.imshow("photo", th4)
cv2.putText(image, text, (int(tl[0]), int(tl[1]) - 20),
		cv2.FONT_HERSHEY_SIMPLEX, thick/8, (0, 0, 255), (int)(thick/4))
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img', 800, 600)
cv2.imshow("img", image)

cv2.waitKey(0)
cv2.destroyAllWindows()