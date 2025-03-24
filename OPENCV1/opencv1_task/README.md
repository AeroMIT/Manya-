## OPENCV1_REPORT

Code:
    #FINAL CODE
    #bounding box (purple + red) + text

    import cv2
    import numpy as np
    
    img = cv2.imread('fruit.jpg', cv2.IMREAD_COLOR)
    img = cv2.resize(img, (900,900))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    kernel = np.ones((2,2), np.uint8)
    
    #for purple
    lower_purple = np.array([126, 150, 83])
    upper_purple = np.array([137, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_purple, upper_purple)

    erosion_p = cv2.erode(mask1, kernel, iterations = 1)
    dilation_p = cv2.dilate(erosion_p, kernel, iterations = 2)
    result_p = cv2.bitwise_and(img, img, mask = dilation_p)
    
    #bounding box for purple with text
    contours_p, _ = cv2.findContours(dilation_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_p:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x-7, y-7), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Purple", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    
    #for red
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    dilation_r = cv2.dilate(mask2, kernel, iterations = 2)
    result_r = cv2.bitwise_and(img, img, mask = dilation_r)
    
    mask = mask1 + mask2
    final_mask = dilation_r + dilation_p

    result = cv2.bitwise_and(img, img, mask = final_mask)

    #bounding box around red with text
    contours_r, _ = cv2.findContours(dilation_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 10
    max_area = 2000

    for cnt in contours_r:
    area = cv2.contourArea(cnt)

        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (200, 0, 255), 2)
            cv2.putText(img, "Red", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 255),2)

    cv2.imshow('result', result)
    cv2.imshow('original', img)
    #cv2.imshow('final mask', final_mask)
    #cv2.imshow('result_p', result_p)
    #cv2.imshow('result_r', result_r)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

 
REPORT:
ORIGINAL IMG: 
![fruit](https://github.com/user-attachments/assets/ca54fcd4-a823-46f3-9f7d-b81c4ad5c280)

Import cv2: OpenCV library used for image processing
Import numpy as np: This library handles arrays
img = cv2.imread('fruit.jpg', cv2.IMREAD_COLOR): reading the image
 
img = cv2.resize(img, (900,900)): resizing the image to 900*900 pixels
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV): converts the image from bgr (default) to hsv; HSV is better for colour detection.
kernel = np.ones((2,2), np.uint8): it creates a 2x2 matrix used for morphological operations like erosion and dilation. The operations help to remove noise from the detected colours.

DETECTING PURPLE FRUITS-
lower_purple = np.array([126, 150, 83])
upper_purple = np.array([137, 255, 255])
it defines a range for the hsv values, the lower value and the upper value for colour detection. This particular range gives us purple.
mask1 = cv2.inRange(hsv, lower_purple, upper_purple):
 
this creates a binary mask where the pixels that fall in the range are white (255) and out of the range are black (0). This isolates the purple areas in the image.
In this mask we see some noise getting detected apart from the fruit.
to remove that we use erosion and dilation.
erosion_p = cv2.erode(mask1, kernel, iterations = 1) : this removes the small noise by shrinking the white areas. Erosion is applied on the mask1 and it iterates once
dilation_p = cv2.dilate(erosion_p, kernel, iterations = 2): this expands the white areas again, restroing the main detected region while removing the noise. It iterates twice. This is applied on the erosion_p.
result_p = cv2.bitwise_and(img, img, mask = dilation_p)
 
this keeps only the purple-coloured region as defined in the dilation_p and replaces everything else with black.
contours_p, _ = cv2.findContours(dilation_p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.findContours: this finds the outline of the detected objects in dilation_p (mask)
cv2.RETR_EXTERNAL: it retrives only the outermost contours (there can be inner ones as well)
cv2.CHAIN_APPROX_SIMPLE: it simplifies the contours to save in memory

for cnt in contours_p:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x-7, y-7), (x + w, y + h), (0, 255, 0), 2)
cv2.boundingRect(cnt): this finds the smallest rectangle that can bound the detected purple fruit.

It loops the detected contours (contours_p), which contains all the detected purple fruits.
Finds the bounding box.
 cv2.boundingRect(cnt) determines the smallest rectangle that can contain the detected object. 
x, y → Top-left corner of the bounding box.
w, h → Width and height of the bounding box.
(0, 255, 0): The colour (green) in BGR format.
x-7, y-7 makes the rectangle slightly bigger
    cv2.putText(img, "Purple", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
Put a label "Purple" near the object
(x, y-10) places the text slightly higher.
cv2.FONT_HERSHEY_SIMPLEX: font style
0.7: font size
(0, 255, 0): Text colour (green).
 2: Thickness


After this the image looks like this:
 



DETECTING RED FRUITS:

lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
this is the range for detecting the red color in the given image in HSV

mask2 = cv2.inRange(hsv, lower_red, upper_red)
like mask1, this creates a binary mask where only the pixels in the given range (lower_r and upper_r) are white and the rest are black.

dilation_r = cv2.dilate(mask2, kernel, iterations = 2)
this explands the detected red areas and fills the small gaps (noise).

mask = mask1 + mask2
final_mask = dilation_r + dilation_p
combining both the masks. In mask we get the mask1 and mask2 combined without the morphological operations and final_mask gives the mask after the morphological operations.
 
result_r = cv2.bitwise_and(img, img, mask = dilation_r)
this, similar to purple, extracts the red region from the region

contours_r, _ = cv2.findContours(dilation_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.findContours(): finds the edges of the object in the image
cv2.RETR_EXTERNAL: retrives only the outer most contours (ignores the inner details)
contours_r: This is a list of all detected object outlines (contours).

min_area = 10
max_area = 2000
filters out the area ranging from the min_area to max_area
for cnt in contours_r: This loop goes through each detected object (contour) one by one. Cnt Represents a single detected object in the image.

area = cv2.contourArea(cnt)
cv2.contourArea(cnt): Measures the total size (area) of the detected object.
area: Stores the size of the current object.

if min_area < area < max_area: filter outs the area based on size.

x, y, w, h = cv2.boundingRect(cnt): Creates a rectangle around the detected object.
 

FINAL IMAGE:
![Screenshot 2025-03-22 200111](https://github.com/user-attachments/assets/95a976f7-1280-47bc-ad1f-ab26484b890c)
