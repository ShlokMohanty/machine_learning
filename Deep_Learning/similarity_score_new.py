from skimage.metrics import structural_similarity 
import cv2
import matplotlib.pyplot as plt 

def orb_sim(img1, img2):
  orb = cv2.ORB_create()
  
  #detect keypoints and descriptors
  kp_a, desc_a = orb.detectAndCompute(img1, None) ##keypoints and descriptors finders 
  kp_b, desc_b = orb.detectAndCompute(img2, None) #keypoints and descriptors finders 
  
  #define the bruteforce matcher object 
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(desc_a, desc_b) #matches
  
  #sort them based on match distance
  matches = sorted(matches[:21], key=lambda x: x.distance)

  #Extract mecth scores 
  #match_scores = [match.distance for match in matches[:21]]
  
  #print("u", match_scores)
  
  final_img = cv2.drawMatches(img1, kp_a, img2, kp_b, matches[:21], None)
  cv2.imshow(final_img)
  for keypoint in kp_a:
    print(keypoint.angle)

  for des in desc_a:
    print(des)
   print("Keypoints 1st Image: " + str(len(kp_a)))
   print("Keypoints 2nd Image: " + str(len(kp_b)))
   similar_regions = [i for i in matches if i.distance < 70]
   if len(matches) == 0:
     return 0
    return len(similar_regions) / len(matches)
    single_match = matches[0]
    single_match.distance 
    matches = sorted(matches, key=lambda x: x.distance)

    #Extract match scores 
    match_scores = [match.distance for match in matches[:21]]

    #plot histogram
    plt.hist(match_scores, bins=50, range=[0, 100])
    plt.title('Histogram of Match scores')
    plt.xlabel('Match Score')
    plt.ylabel('Frequency')
    plt.show()

def structural_sim(img1, img2, full=True):

  sim, diff = structural_similarity(img1, img2, full=True)
  return sim
img00 = cv2.imread()
img01 = cv2.imread()
img1 = cv2.resize()
img2 = cv2.resize()

orb_similarity = orb_sim(img1, img2)
print("Similarity using ORB is: ", orb_similarity*100)
from skimage.transform import resize 
ssim = structural_sim(img1, img2)
print("Similarity using SSIM is: ", ssim*100)
cv2.waitKey(0)
cv2.destroyAllWindows()

 
  
