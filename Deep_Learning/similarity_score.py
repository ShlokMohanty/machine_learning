from skimage.metrics import structural_similarity 
import cv2

def ORB_sim(img1, img2):
  orb = cv2.ORB_create()
  kp_a, desc_a = orb.detectAndCompute(img1, None)
  kp_b, desc_b = orb.detectAndCompute(img2, None)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  mathes = bf.match(desc_a, desc_b)
  match_scores = [match.distance for match in matches[:50]]
  print("u", match_scores)
  final_img = cv2.drawMatches(img1, kp_a, img2, kp_b, matches[:50], None) #drawing almost 50 matches 
  cv2.imshow("Matches", final_img)
  similar_regions = [i for i in matches if i.distance < 50] #iterating over the matches when the distance is 50
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)

def structural_sim(img1, img2):
  sim, diff = structural_similarity(img1, img2, full=True)
  return sim #for returning the similarities

img00 = cv2.imread()
img1 = cv2.resize() #resized image of the original image 1
img01 = cv2.imread()
img2 = cv2.resize() #resized image of original image 2
orb_similarity = orb_sim(img1, img2)
from skimage.transform import resize
ssim = structural_sim(img1, img2)
cv2.waitkey(0)
cv2.destroyAllWindows()
  
