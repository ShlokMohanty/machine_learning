import os 
import cv2
def compute_orb_score(image1, image2):
  orb = cv2.ORB_create()
  kp1, des1 = orb.detectAndCompute(image1, None)
  kp2, des2 = orb.detectAndCompute(image2, None)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
  matches = bf.match(des1, des2)
  orb_score = sum([match.distance fro match in matches] / max(len(matches), 1))
  return orb_score 
def compute_similarity_score(orb_score):
  similarity_score = 1 - (orb_score / 100)
  return similarity_score 
def compare_folders(folder1, folder2):
  images_1 = [cv2.imread(os.path.join(folder1, filename), cv2.IMREAD_GRAYSCALE) for filename in os.listdir(folder1)]
  images_2 = [cv2.imread(os.path.join(folder2, filename), cv2.IMREAD_GRAYSCALE) for filename in os.listdir(folder2)]
  results = []
  for image_1 in images_1:
    for image_2 in images_2:
      orb_score = compute_orb_score(image_1, image_2)
      similarity_score = compute_similarity_score(orb_score)
      results.append((image_1, image_2, orb_score, similarity_score))
  return results

def visualize_results(results):
  for idx, (image_1, image_2, orb_score, similarity_score) in enumerate(results):
    cv2.imshow(f"Image 1 - {idx}", image_1)
    cv2.imshow(f"Image 2 - {idx}", image_2)
    print(f"ORB Score: {orb_score}", Similarity Score: {similarity_score})
    cv2.waitKey(0)
    cv2.destroyAllWindows()

folder1 = ""
folder2 = ""
results = compare_folders(folder1, folder2)
visualize_results(results)

  
