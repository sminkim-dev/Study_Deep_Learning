# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append('..')
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

imgs = x_test
labels = t_test
print('학습  데이터수 : ', x_test.shape[0])
print('테스트데이터수 : ', t_test.shape[0])
imgs = imgs.reshape((10000,28,28))

SIZE = imgs.shape[0] * imgs.shape[1] * imgs.shape[2]
COL = 250 * imgs.shape[1]
ROW = SIZE // COL

for i in range(0, ROW, 28):
    new_img = imgs[i]
    for j in range(1+i,(COL//28)+i):
        new_img = np.concatenate((new_img,imgs[j]),axis=1)
    if i == 0:
        new_img2 = new_img
    else:
        new_img2 = np.concatenate((new_img2,new_img))

print(new_img2.shape)

img_show(new_img2)
input()