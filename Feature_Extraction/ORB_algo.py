import numpy as np 
import cv2 
query_img = cv2.imread("") #reading the query img 
train_img = cv2.imread("") #reading the training img 
query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY) #grayscale img conversion for query img
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY) #grayscale img conversion for train img
orb = cv2.ORB_create() #orb model creation or initialization 
query_points, queryDescriptors = orb.detectAndCompute(query_img_bw, None) #defining the descriptors and key points
trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None) 
matcher = cv2.BFMatcher() 
matches = matcher.match(queryDescriptors, trainDescriptors) #for matching score 
final_img = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, matches[:20], None) #finding out the matches 
final_img = cv2.resize(final_img, (1000, 650)) #resizing the images 
cv2.imshow("Matches", final_img)
cv2.waitKey(3000)
