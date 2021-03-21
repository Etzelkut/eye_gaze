from torchvision import transforms
import torch
import numpy as np
import cv2
import PIL.Image

from model import Re_pl
from irislandmarks import IrisLandmarks
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Connect_module():
  def __init__(self, path_cascade, path_dots, path_eye_model):
    self.face_cascade = cv2.CascadeClassifier(path_cascade)
    self.transforms = transforms.Compose([transforms.Resize((224, 224)), 
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
                                        )
    self.dots_model = Re_pl.load_from_checkpoint(path_dots).to(device)
    self.dots_model.eval()
    self.dots_model.freeze()
    self.eye_model = IrisLandmarks().to(device)
    self.eye_model.load_weights("irislandmarks.pth")
  
  def forward_dots(self, image):
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = self.face_cascade.detectMultiScale(image1, 1.1, 4)
    if len(faces) != 0:
      (x, y, w, h) = faces[0]
    else:
      (x, y, w, h) = (0, 0, 400, 400)
    
    if y+h >= image1.shape[0]:
      max_y = image1.shape[0] - 1
    else:
      max_y = y+h
    
    if x+w >= image1.shape[1]:
      max_x = image1.shape[1] - 1
    else:
      max_x = x+w
    
    new_img = image1[y:max_y, x:max_x]
    new_img = PIL.Image.fromarray(new_img)
    new_img = self.transforms(new_img).to(device)
    landmarks = self.dots_model(new_img.unsqueeze(0))
    landmarks = (landmarks.view(68,2).cpu().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
    eyes_pos = np.copy(landmarks[36:48])

    left = eyes_pos[0:6]
    right = eyes_pos[6:12]
    eyes_pos, left, right
    l_mean = np.mean(left, axis=0)
    r_mean = np.mean(right, axis=0)

    return image1, np.round(landmarks).astype(int), l_mean, r_mean, left, right

  def forward_eye(self, image, l_mean, r_mean):
    poslr = np.zeros((2, 2))
    poslr[0] = l_mean
    poslr[1] = r_mean

    poslr = np.round(poslr - [50, 50]).astype(int)
    lll = poslr[0]
    rrr = poslr[1]
    left_eye_im = image[lll[1]-30:lll[1]+120, lll[0]:lll[0]+100]
    right_eye_im = image[rrr[1]-30:rrr[1]+120, rrr[0]:rrr[0]+100]

    img = np.zeros((2, 64, 64, 3))

    #img[0] = cv2.resize(left_eye_im, (64, 64))
    #img[1] = cv2.resize(right_eye_im, (64, 64))

    try:
      img[0] = cv2.resize(left_eye_im, (64, 64))
      img[1] = cv2.resize(right_eye_im, (64, 64))

    except Exception as e:
      print(str(e))

    _, iris_gpu = self.eye_model.predict_on_batch(img)
    iris = iris_gpu.cpu().numpy()

    return iris, lll, rrr

  def forward(self, image1):
    image, landmarks, l_mean, r_mean, left, right = self.forward_dots(image1)

    def get_new(left):
      new_left = np.zeros((4, 2))
      new_left[0], new_left[1], new_left[2], new_left[3] = left[0], np.mean(left[1:3], axis = 0), left[3], np.mean(left[4:6], axis = 0)
      return new_left

    new_left = get_new(left)
    new_right = get_new(right)

    iris, lll, rrr = self.forward_eye(image, l_mean, r_mean)

    w, h = 100/64, 150/64

    for (y, x) in landmarks:
      image[x-2:x+2, y-2:y+2] = [255, 0, 0]
    
    x, y = iris[0][:, 0], iris[0][:, 1]
    x = x * w  + lll[0] - 1
    y = y * h  + lll[1] - 31

    for i in range(5):
      x1 = np.round(x[i]).astype(int)
      y1 = np.round(y[i]).astype(int)
      image[y1-2:y1+2, x1-2:x1+2] = [0, 255, 0]

    def get_newxy(x, y, new_left):
      to_left = x - new_left[0][0]
      to_right = x - new_left[2][0]
      to_top = y - new_left[1][1]
      to_bottom = y - new_left[3][1]

      new_x = (to_right + to_left) * 2
      new_y = (to_top + to_bottom)

      att_x = int(x + new_x) #!
      att_y = int(y + new_y) #!

      return att_x, att_y
    
    att_x, att_y = get_newxy(x[0], y[0], new_left) #!
    dis1, dis2 = y[4] - y[2], new_left[0][0] - new_left[2][0] #!
    dis2 = int(dis2) #!

    blank = np.zeros((100, 100, 3))
    red = (dis2**2)/125 * 255
    blue = (128 - dis2) if dis2 < 128 else 0
    blank[:, :, 0:3] = [red, 0, 0]
    image[0:100, 0:100] = blank

    x, y = iris[1][:, 0], iris[1][:, 1]
    x = x * w  + rrr[0] - 1
    y = y * h  + rrr[1] - 31
    
    for i in range(5):
      x1 = np.round(x[i]).astype(int)
      y1 = np.round(y[i]).astype(int)
      image[y1-2:y1+2, x1-2:x1+2] = [0, 255, 0]
    
    image[att_y-10:att_y+10, att_x-10:att_x+10] = [0, 0, 255]
    #att_x, att_y = get_newxy(x[0], y[0], new_right) #!
    #image[att_y-5:att_y+5, att_x-5:att_x+5] = [0, 0, 255]
    return image
