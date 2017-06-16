import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
kernel = np.ones((5, 5), np.uint8)

b, g, r = cv2.split(frame)
close_b = cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel)
close_b = cv2.threshold(close_b, 127, 255, cv2.THRESH_BINARY)[1]
close_g = cv2.morphologyEx(g, cv2.MORPH_OPEN, kernel)
close_g = cv2.threshold(close_g, 127, 255, cv2.THRESH_BINARY)[1]
close_r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel)
close_r = cv2.threshold(close_r, 127, 255, cv2.THRESH_BINARY)[1]

clone_close_b = close_b.copy()
clone_close_g = close_g.copy()
clone_close_r = close_r.copy()
im2_b, contours_b, hierarchy_b = cv2.findContours(clone_close_b, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
im2_g, contours_g, hierarchy_g = cv2.findContours(clone_close_g, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
im2_r, contours_r, hierarchy_r = cv2.findContours(clone_close_r, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

bounding_boxes = []
contours = contours_b + contours_g + contours_r

for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    bounding_boxes.append((x, y, w, h))
    cv2.rectangle(frame,(x,y),(x+w, y+h), (0, 255, 0), 2)
    cv2.circle(frame, (x+w/2, y+h/2), 3, (255, 0, 0), 3)

ROI_HIST = []
for bounding_box in bounding_boxes:
    roi = frame[bounding_box[1]:bounding_box[1]+bounding_box[3],
          bounding_box[0]:bounding_box[0]+bounding_box[2]]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    ROI_HIST.append(roi_hist)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(True):
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



    index = 0
    for hist in ROI_HIST:
        dst = cv2.calcBackProject([hsv], [0], hist, [0, 180], 1)
        bounding_boxes[index] = cv2.CamShift(dst, bounding_boxes[index], term_crit)[1]
        cv2.circle(frame, (bounding_boxes[index][0] + bounding_boxes[index][2] / 2,
                           bounding_boxes[index][1] + bounding_boxes[index][3] / 2), 3, (0, 255, 255), 3)

        index += 1

    cv2.imshow('frame', frame)
    k = cv2.waitKey(60) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
