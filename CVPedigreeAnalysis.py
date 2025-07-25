import pytesseract
import cv2
import re
from pprint import pprint
import math
import numpy as np

def linDist(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2

    return((x2-x1)**2+(y2-y1)**2)**0.5

def closestLabel(marker_coords, label_dict):
    closest_distance = math.inf
    closestLabel = None
    for label in label_dict.keys():
        label_x, label_y, label_w, label_h = label_dict[label]
        label_coords = ((label_x + label_w/2), (label_y + label_h/2))
        distance = linDist(marker_coords, label_coords)
        if distance < closest_distance:
            closest_distance = distance
            closestLabel = label
    return closestLabel

def merge_duplicate_lines(lines, angle_threshold= 10, distance_threshold= 50):
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        duplicate = False
        for merged_line in merged_lines:
            mx1, my1, mx2, my2 = merged_line
            line_angle = np.arctan2((y2-y1), (x2-x2)) 
            merged_line_angle = np.arctan2((my2-my1), (mx2-mx2)) 
            angle_difference = abs(np.degrees(line_angle - merged_line_angle))

            if angle_difference < angle_threshold and linDist((x1,y1), (mx1,my1)) < distance_threshold:
                duplicate = True
                #breaks loop through merged lines since we determined it to be duplicate
                break
        if not duplicate:
            merged_lines.append(line[0])
    
    return merged_lines


def categorize_normalize_lines(lines):
    categorized_normalized_lines = {
    'vertical': [],
    'horizontal': []
    }
    print(f'Number of Lines Detected: {len(lines)}')
    for line in lines:
        x1,y1,x2,y2 = line
        
        if x1 == x2:
            if y1 < y2:
                z = y1
                y1 = y2
                y2 = z
            categorized_normalized_lines['vertical'].append((x1, y1, x2, y2))
        elif y1 == y2:
            if x1 > x2:
                z = x1
                x1 = x2
                x2 = z                
            categorized_normalized_lines['horizontal'].append((x1, y1, x2, y2))

    return categorized_normalized_lines

def trackRelation(normalized_categorized_lines, IndvDataDict, distance_threshold = 25):
    connection_lines = []
    starting_coords = []
    #find initional starting coordinates
    for IndvID in IndvDataDict.keys():
        x_left, y_top, w, h = IndvIDsDict[IndvID]
        x_mid = x_left + w/2
        top_center_coord = (x_mid, y_top)
        for vert_line in normalized_categorized_lines['vertical']:
            Vx1,Vy1,Vx2,Vy2 = vert_line
            if linDist(top_center_coord, (Vx1,Vy1)):
                starting_coords.append((Vx1,Vy1))

    for s_coord in starting_coords:
        #find initial veritcal line
        for vert_line in normalized_categorized_lines['vertical']:
            Vx1,Vy1,Vx2,Vy2 = vert_line
            if linDist(s_coord, (Vx1, Vy1)) < distance_threshold:
                current_coord = (Vx2, Vy2)
                break
        
        #find initial horizontal
        for horz_line in normalized_categorized_lines['horizontal']:
            Hx1,Hy1,Hx2,Hy2 = horz_line
            Cx,Cy = current_coord
            if abs(Cy - Hy1) < distance_threshold and (Hx1 < Cx and Cx < Hx2):
                endpoints = horz_line
                break
        
        #find secondary vertical (if it exists)
        next_coord= 0
        for vert_line in normalized_categorized_lines['vertical']:
            Vx1,Vy1,Vx2,Vy2 = vert_line
            Cx1,Cy1,Cx2,Cy2 = endpoints
            if abs(Cy1 - Vy1) < distance_threshold and (Cx1 < Vx1 and Vx1 < Cx2):
                next_coord = (Vx2, Vy2)
                break
        
        #check if secondary vertical was found
        if next_coord:
            current_coord = next_coord
            #find secondary horizontal
            for horz_line in normalized_categorized_lines['horizontal']:
                Hx1,Hy1,Hx2,Hy2 = horz_line
                Cx,Cy = current_coord
                if abs(Cy - Hy1) < distance_threshold and (Hx1 < Cx and Cx < Hx2):
                    endpoints = horz_line
                    break
        
    
        fx1,fy1,fx2,fy2 = endpoints
        sx, sy = s_coord
        connection_lines = connection_lines + [(sx,sy,fx1,fy1), (sx,sy,fx2,fy2)]
    return connection_lines

            
                


                


         


    

#----------------------------------------
# INDIVIDUAL ID DETECTION
#----------------------------------------

img = cv2.imread('data/Pedigree1.png')
img_height, img_width, _ = img.shape
img_area = img_height * img_width
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


redacted_img = np.copy(gray_img)
TextData = pytesseract.image_to_data(gray_img, output_type= pytesseract.Output.DICT)
IndvIDsDict = {}
n_boxes = len(TextData['text'])

for i in range(n_boxes):
    if len(TextData['text'][i]) > 0 and TextData['text'][i][0] != ' ':
        IndvID = TextData['text'][i]
        (x,y,w,h) = (TextData['left'][i], TextData['top'][i], TextData['width'][i], TextData['height'][i])
        display_coords = (x,y,w,h)
        IndvIDsDict[IndvID] = display_coords
        #draw a white rectangle over the ID number (with a little extra size for buffer)
        redacted_img = cv2.rectangle(redacted_img, (x-10,y-10), (x+w+20, y+h+20), (255,255,255), -1)

print('Individuals Coordinates')
pprint(IndvIDsDict)

#----------------------------------------
# PHENOTYPE AND SEX DETECTION
#----------------------------------------
annotated_img = np.copy(gray_img)

_, threshold_light = cv2.threshold(redacted_img, 250, 255, cv2.THRESH_BINARY)
_, threshold_dark = cv2.threshold(redacted_img, 5, 255, cv2.THRESH_BINARY_INV)

light_contours, _ = cv2.findContours(threshold_light, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
dark_contours, _ = cv2.findContours(threshold_dark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

Phenotypes = ['+', '-']
for phenotype in Phenotypes:
    contours = light_contours if phenotype == '-' else dark_contours
    for i, contour in enumerate(contours):

        epsilon =0.01 * cv2.arcLength(contour, closed= True)
        approx = cv2.approxPolyDP(contour, 1, True)
        
        x,y,w,h = cv2.boundingRect(approx)
        bounding_area = w*h
        if bounding_area < 0.25*img_area and bounding_area > 0.0001*img_area:

            

            redacted_img = cv2.rectangle(redacted_img, (x-10,y-10), (x+w+20, y+h+20), (255,255,255), -1)

            center_coords = ((x + w/2), y + h/2)
            label = closestLabel(marker_coords= center_coords, label_dict= IndvIDsDict)

            xmid = int(x - 2*w)
            ybelow = int(y + 5*h/2)

            display_coords = (xmid, ybelow)
            colour = (0,0,0)
            font = cv2.FONT_HERSHEY_DUPLEX

            if len(approx) < 10:
                cv2.putText(annotated_img, label + ' ' + phenotype + ' male ', display_coords, font, 1, colour, 1)
            else:
                cv2.putText(annotated_img, label + ' ' + phenotype+ ' female ', display_coords, font, 1, colour, 1)


#----------------------------------------
# RELATION LINE DETECTION
#----------------------------------------
edges = cv2.Canny(redacted_img, 250, 255)
raw_lines = cv2.HoughLinesP(edges,
                        lines= np.array([]),
                        rho=1, 
                        theta= np.pi/180,
                        threshold= 100,
                        minLineLength= 50,
                        maxLineGap= 50)
line_img = np.copy(annotated_img)*0
lines = merge_duplicate_lines(raw_lines)
cat_norm_lines = categorize_normalize_lines(lines)

trial_s_coord = (569,1221)
connection_lines = trackRelation(cat_norm_lines, IndvIDsDict)

for line in lines:
    x1, y1, x2, y2 = line
    line_img = cv2.line(line_img, (x1,y1), (x2,y2), (255,255,255), 5)

for line in connection_lines:
    x1, y1, x2, y2 = line
    line_img = cv2.line(line_img, (x1,y1), (x2,y2), (255,255,255), 5)


print('Normalized Connection Coordinates')
pprint(cat_norm_lines)
       







# annotated_img = cv2.addWeighted(annotated_img, 0.8, line_img, 0.2, 0)

cv2.imshow('redacted', redacted_img)
cv2.imshow('lines', line_img)
cv2.imshow('annotated', annotated_img)
k = cv2.waitKey(0)
if k == ord('s'):
    cv2.imwrite('data/Pedigree1AutoRedacted.png', img)
cv2.destroyAllWindows