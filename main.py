import cv2
import numpy as np 

# Load the image
image = cv2.imread("1.JPG")
orig = image.copy()

# Step 1, find edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 35, 200)

print "STEP 1: Edge Detection"
cv2.imwrite('step1.jpg', edged)

# Step 2, find countours and sort in order of the area. 
# We assume the card is the focus of the picture so it should have the largest area
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
# loop over the contours
screenCnt = None
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if our approximated contour has four points, then we
	# can assume that we have found our card
	if len(approx) == 4:
		screenCnt = approx
		break
 
# show the contour (outline) of the card
print "STEP 2: draw contours of card"
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imwrite('step2.jpg', image)

#Step 3
# Map the found coordinates of the countour into a 483x300 image
destPoints = np.array([ [0,0],[482,0],[482,299],[0,299] ],np.float32)
sourcePoints = np.array([ screenCnt[3][0], screenCnt[2][0], screenCnt[1][0], screenCnt[0][0] ], np.float32);
transform = cv2.getPerspectiveTransform(sourcePoints, destPoints)
warp = cv2.warpPerspective(image, transform ,(483,300))
cv2.imwrite('step3.jpg', warp)