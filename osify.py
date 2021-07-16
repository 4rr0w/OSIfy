import cv2
import imutils
import numpy as np
import csv
import pytesseract
import argparse
from huepy import *
from pytesseract.pytesseract import Output #refer to this https://github.com/s0md3v/huepy

parser = argparse.ArgumentParser()
parser.add_argument('-p', help = "Path to image", dest = 'img_path', default ='none', required = True)
parser.add_argument('-v', help = "Flag for vechile plate text extraction", dest = 'vehicle_plate', default = False, action = 'store_true')
parser.add_argument('-a', help = "Flag to turn off full text extraction", dest = 'all_text', default = True, action = 'store_false')

args = parser.parse_args() 
img_path = args.img_path
vehicle_plate = args.vehicle_plate
all_text = args.all_text

class AnalyseImage:
    def extract_Vechical_Number(self, gray, img):
        edged = cv2.Canny(gray, 30, 200) 
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        screenCnt = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        
            if len(approx) == 4:
                screenCnt = approx
                break
        if screenCnt is None:
            detected = 0
            print(bad(red("No contour detected")))
        else:
            detected = 1

        if detected == 1:
            cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

        mask = np.zeros(gray.shape,np.uint8)
        new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
        new_image = cv2.bitwise_and(img,img,mask=mask)

        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx:bottomx+1, topy:bottomy+1]

        text = pytesseract.image_to_string(Cropped, config='--psm 11')
        print(good(green("Detected license plate Number is: "+ text)))


    def extract_Any_Text(self, gray_img):
        
        threshold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        custom_config = r'--oem 3 --psm 6'
        details = pytesseract.image_to_data(threshold_img, output_type=Output.DICT, config=custom_config, lang="eng")
        total_boxes = len(details['text'])

        for sequence_number in range(total_boxes):
            if int(details['conf'][sequence_number]) >30:
                (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
                threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        parse_text = []
        word_list = []
        last_word = ''   
        for word in details['text']:     
            if word!='':
                word_list.append(word)
                last_word = word
            if (last_word!='' and word == '') or (word==details['text'][-1]):
                parse_text.append(word_list)
                word_list = []
            with open("text.csv", "w", newline="") as file:
                csv.writer(file, delimiter=" ").writerows(parse_text)
            
        print(good("Extracted text saved to ./text.csv"))

def main():
    analyser = AnalyseImage()
    try:
        img_original = cv2.imread(img_path,cv2.IMREAD_COLOR)
        img = cv2.resize(img_original, (600,400) )
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        gray = cv2.bilateralFilter(gray_img, 13, 15, 15) 

        if(all_text):
            analyser.extract_Any_Text(gray)
        if(vehicle_plate):
            analyser.extract_Vechical_Number(gray_img, img)

    except:
        print(bad(red("Some error occured")))

  
# Using the special variable 
# __name__
if __name__=="__main__":
    main()


