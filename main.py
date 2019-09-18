import cv2
from ocr import get_text_locations
import pytesseract
import sys

img = cv2.imread(sys.argv[1])
img = cv2.resize(img, (1920, 960))
locations = get_text_locations(img, width=1920, height=960)

for location in locations:
	roi = img[location[1]:location[3], location[0]:location[2]]
	config = ("-l eng --oem 1 --psm 7")
	text = pytesseract.image_to_string(roi, config=config)

	# display the text OCR'd by Tesseract
	print("OCR TEXT")
	print("========")
	print("{}\n".format(text))
 
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV, then draw the text and a bounding box surrounding
	# the text region of the input image
	text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
	output = img.copy()
	cv2.rectangle(output, (location[0], location[1]), (location[2], location[3]),
		(0, 0, 255), 2)
	cv2.putText(output, text, (location[0], location[1] - 20),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
 
	# show the output image
	cv2.namedWindow('Text Detection', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('Text Detection', 1200, 600)
	cv2.imshow("Text Detection", output)
	cv2.waitKey(0)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1200, 600)
cv2.imshow("image", img)
cv2.waitKey()
cv2.destroyAllWindows()