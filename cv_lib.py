import cv2
import numpy as np 
import os 
import pytesseract
from PIL import Image
from PIL import ImageFilter
from StringIO import StringIO
from matplotlib import pyplot as plt

MAX_CARDS=6

def show_img(img): 
  while(True):
    cv2.imshow('-1',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()


def find_cards(image):
  # Step 1, find edges
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (5,5), 0)
  edged = cv2.Canny(gray, 35, 200)

  # Step 2, find contours and sort in order of the area. 
  # We assume the card is the focus of the picture so it should have the largest area
  (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:MAX_CARDS*2]
  
  # loop over the contours
  filtered_cnts = []
  for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
   
    # if our approximated contour has four points, then we
    # can assume that we have found our card
    if len(approx) == 4:
      filtered_cnts.append(approx)
  
  double_filtered = []
  for c in filtered_cnts:
    if c[1][0][1] > 600:
      double_filtered.append(c)

  # sort by x value 
  sorted_cnts = sorted(double_filtered, key=lambda x: x[1][0][0])
 
  def contours(screenCnt):
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    try:
      # Map the found coordinates of the contour into a 483x300 image
      destPoints = np.array([ [0,0],[482,0],[482,299],[0,299] ],np.float32)
      sourcePoints = np.array([ screenCnt[3][0], screenCnt[2][0], screenCnt[1][0], screenCnt[0][0] ], np.float32);
      transform = cv2.getPerspectiveTransform(sourcePoints, destPoints)
      warp = cv2.warpPerspective(image, transform, (483,300))
      return warp

    except Exception as e:
      print e 
      return None

  cards = map(contours, sorted_cnts)
  # show_img(image)
  
  return cards 

def rotate(mat, angle):
  height, width = mat.shape[:2]
  image_center = (width/2, height/2)

  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

  abs_cos = abs(rotation_mat[0,0])
  abs_sin = abs(rotation_mat[0,1])

  bound_w = int(height * abs_sin + width * abs_cos)
  bound_h = int(height * abs_cos + width * abs_sin)

  rotation_mat[0, 2] += bound_w/2 - image_center[0]
  rotation_mat[1, 2] += bound_h/2 - image_center[1]

  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat

def rotate_90(mat):
  return rotate(mat, 90)

def find_text_in_top_corner(image):
  # crop image 
  cropped_image = image[0:90, 0:55]
  # im_gray = cv2.imread('grayscale_image.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

  # thresh = 127
  # im_bw = cv2.threshold(cropped_image, thresh, 255, cv2.THRESH_BINARY)[1]
  # gray_image = cv2.cvtColor(im_bw, cv2.COLOR_BGR2GRAY)

  gray_image = cropped_image

  cv2.imwrite("temp/image.png", gray_image)
  text = pytesseract.image_to_string(Image.open('temp/image.png'), lang="Card", config="-c tessedit_char_whitelist=23456789AKJQ  -psm 10")
  return gray_image, text

def find_text(image):
  cv2.imwrite("temp/image.png", image)
  return pytesseract.image_to_string(Image.open('temp/image.png'), lang="Card", config="-c tessedit_char_whitelist=123456789  -psm 10")

def to_bw(image):
  thresh = 127
  bw_im = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
  return cv2.cvtColor(bw_im, cv2.COLOR_BGR2GRAY)

# def show_images(images):

# crop based on fixed poitns 
# given big image

# get images in photos 
# sort based on x 

# filtered_imgs = []
# for each image 
  # get thte top corner 
  # run tess
  # if nil
    # get the flipped
    # get thte top corner 
    # run tess
    # if nill 
      # continue 
  # add to  filtered imgs array 

# print! 
def get_cards(path):
  # load original 
  # path = os.path.join("temp","20170226_064440.jpg")
  image = cv2.imread(path)  

  cards = find_cards(image)
  rotated_cards = map(rotate_90, cards)

  cards = []
  for card in rotated_cards:
    top_corner = card[0:90, 0:55]
    bw_im = to_bw(top_corner)

    # print i, np.sum(gray_image)
    text = ""
    # is top corner all white? 
    if np.sum(bw_im) > 1200000:
      # flip image
      card = cv2.flip(card,1)
      top_corner = card[0:90, 0:55]
      bw_im = to_bw(top_corner)

    text = find_text(bw_im) 
    if text == "":
      card = rotate(card, 180)
      top_corner = card[0:90, 0:55]
      bw_im = to_bw(top_corner)
    text = find_text(bw_im) 

    print text

    import uuid
    uid = str(uuid.uuid1())
    cv2.imwrite("static/cards/"+uid+'.png', card)
    cv2.imwrite("temp/out/"+uid+'-top.png', bw_im)

    c = {
      'url': "/cards/"+uid+'.png',
      'val': int(text)
    }
    cards.append(c)
  return cards






# flip = cv2.flip(card,1)
    # cv2.imwrite("temp/out/"+str(i)+'-flip.png', flip)
    # text = None
    # gray_image, text = find_text_in_top_corner(card)
    # if(text == None):
    #   gray_image, text = find_text_in_top_corner(cv2.flip(card,1))
    # print i, text
    # # if(text == ""):
    # cv2.imshow(str(i),card)

    # # else:  
    # #   cv2.imshow(str(i),gray_image)



# while(True):

#   if cv2.waitKey(1) & 0xFF == ord('q'):
#     break



# # def process_image(file):
# #     image = Image.open(file)
# #     image.filter(ImageFilter.SHARPEN)
# #     return pytesseract.image_to_string(image)


# # # import pytesser
# # # set up a video
# # # every five seconds take a photo
# # # save photo to disk
# # # stich photos together 
# # # detect cards 
# # # show cards 



# # # Load the image
# # # image = cv2.imread("2.JPG")
# # # orig = image.copy()

# # image = None

# # def stich_images(images):
# #   stitcher = cv2.create_stitcher(False)
# #   result = stitcher.stitch(tuple(images))
# #   return result[1]


# # def run():
# #   cap = cv2.VideoCapture(0)

# #   while(True):
# #     # Capture frame-by-frame
# #     ret, frame = cap.read()

# #     # Our operations on the frame come here
# #     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     image = frame
# #     images = get_images()

# #     # Display the resulting frame
# #     cv2.imshow('-1',frame)
# #     if(images is not None):
# #       print len(images)
# #       for i in range(len(images)):
# #         cv2.imshow(str(i),images[i])
    
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #       break

# #   # When everything done, release the capture
# #   cap.release()
# #   cv2.destroyAllWindows()

# def show_img(img): 
#   while(True):
#     cv2.imshow('-1',img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#       break

#   # When everything done, release the capture
#   cap.release()
#   cv2.destroyAllWindows()

# # # stich images 
# # def load_images_from_folder(folder):
# #   images = []
# #   for filename in os.listdir(folder):
# #     img = cv2.imread(os.path.join(folder,filename))
# #     if img is not None:
# #       images.append(img)
# #   return images

# # # images = load_images_from_folder('temp/')
# # # pano = stich_images(images)
# # # show_img(pano)

# def rotate(mat):
#   angle = 90
#   height, width = mat.shape[:2]
#   image_center = (width/2, height/2)

#   rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

#   abs_cos = abs(rotation_mat[0,0])
#   abs_sin = abs(rotation_mat[0,1])

#   bound_w = int(height * abs_sin + width * abs_cos)
#   bound_h = int(height * abs_cos + width * abs_sin)

#   rotation_mat[0, 2] += bound_w/2 - image_center[0]
#   rotation_mat[1, 2] += bound_h/2 - image_center[1]

#   rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
#   return rotated_mat

# # def crop(img):
# #   cropped_img = img[2:90, 2:55]
# #   return cropped_img

# # image = cv2.imread(os.path.join("temp","left.png"))
# # images = get_images()
# # images = map(rotate, images)
# # flipped = []
# # for img in images: 
# #   flipped.append(cv2.flip(img,1))

# # images = map(crop, images)
# # flipped = map(crop, flipped)

# # # for img in flipped:
# # #   img = Image.fromarray(img)
# # #   print(pytesseract.image_to_string(img))

# images = cards

# # img = images[0]
# # # im_gray = cv2.imread('grayscale_image.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

# # thresh = 127
# # im_bw = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
# # gray_image = cv2.cvtColor(im_bw, cv2.COLOR_BGR2GRAY)

# # cv2.imwrite("test.png", gray_image)

# # print(process_image("test.png"))

# # # pytesseract.image_to_string(Image.open('test.png'), config="-c tessedit_char_whitelist=0123456789AKJQ  -psm 10")

# # while(True):
# #   cv2.imshow("0",img)
# #   cv2.imshow("1",gray_image)

# # # # # height, width = image.shape[:2]
# # # # print len(images)
# while(True):
#   for i in range(len(images)):
#     cv2.imshow(str(i),images[i])
#     # cv2.imshow(str(i+len(images)),flipped[i])
#   # cv2.imshow("image", image)

#   # for img in images:
#   #   gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#   #   import time

#   #   cv2.imshow(str()),gray_image)
  

#   if cv2.waitKey(1) & 0xFF == ord('q'):
#     break


# def pattern_match(image):
#   img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   template = cv2.imread("temp/spade.jpg")

#   w, h = template.shape[::-1]

#   res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
#   threshold = 0.8
#   loc = np.where( res >= threshold)
#   for pt in zip(*loc[::-1]):
#     print pt
#     # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

#   # print len(zip(*loc[::-1]))