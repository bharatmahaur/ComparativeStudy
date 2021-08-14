import cv2

def ploy_rect(image, box, color):
    ymin, xmin, ymax, xmax = box
    line_thickness = int(round(0.002 * max(image.shape[0:2])) / 5)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=color , thickness=line_thickness)