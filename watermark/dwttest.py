import cv2
import numpy as np
import  pywt
import matplotlib.pyplot as plt

waterimg=cv2.imread('./1.jpg')
waterimg2=cv2.imread('./3.jpg')
waterimg3=cv2.imread('./sprb2.png')
cv2.imshow('t',waterimg2)
#img_gauss = cv2.GaussianBlur(waterimg2, (3,3), 1)
p1=pywt.wavedec2(waterimg2,"bior1.3",level=1)
p2=pywt.wavedec2(waterimg,'bior1.3',level=1)
#p3=pywt.wavedec2(img_gauss,'db2',level=3)
print(np.array(p1[1]).shape)
cv2.imshow('1',p1[1][0])
cv2.imshow('11',p1[1][1])
cv2.imshow('111',p1[1][2])
cv2.imshow('2',p2[1][0])
cv2.imshow('22',p2[1][1])
cv2.imshow('222',p2[1][2])
cv2.waitKey()
def haar_img():
    img_u8 = cv2.imread("./3.jpg")
    img_f32 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)

    plt.figure('二维小波一级变换')
    coeffs = pywt.dwt2(img_f32, 'haar')
    cA, (cH, cV, cD) = coeffs

    # 将各个子图进行拼接，最后得到一张图
    AH = np.concatenate([cA, cH], axis=1)
    VD = np.concatenate([cV, cD], axis=1)
    img = np.concatenate([AH, VD], axis=0)
    return img



img = haar_img()

plt.imshow(img, 'gray')
plt.title('img')
plt.show()
qt=[]
sum1=0
print(np.array(p1[1]).shape)
print(p1[1][1][1][1]," " ,p2[1][1][1][1])
for i in range(len(p1[1])):
    for j in range(len(p1[1][0])):
        for kk in range(len(p1[1][0][1])):
            try:
                q1=p1[1][i][j][kk]-p2[1][i][j][kk]
                if abs(q1[0])+abs(q1[1])+abs(q1[2])>10:
                #print(q1,p1[1][i][j][kk])
                    sum1+=1
                    qt.append(q1)
            except:
                break

print(sum1)
#print(qt)
# cv2.imshow('gs',img_gauss)
# c3=pywt.wavedec2(img_gauss,'db2',level=1)
q=[0]*27
# #cv2.imshow('start',waterimg)
# #c=pywt.dwt()
m=[0]*27
# c2=pywt.wavedec2(waterimg,'db2',level=1)
# print(np.array(qt).shape)
# print(qt[1])
# f=open('test.txt','w')
# for item in qt:
#
#         f.write(str(item))
# f.write("/n")
# f.close()
# c22=pywt.wavedec2(waterimg2,'db2',level=1)
# # ff=open('4.txt','w')

#
# # for item in c22[0]:
# #     for opt in item:
# #         for k in opt:
# #             if k>max1:
# #                 max1=k
# #             if k<min1:
# #                 min1=k
# # print(max1,min1)
min1=1000
max1=-100
min2=1000
max2=-100
p=[0]*27
min3=1000
max3=-100
for item in p1[3][0]:
    for opt in item:
        q[int(opt[0]//45+1)]+=1
for item in p2[3][0]:
    for opt in item:
        m[int(opt[0]//45+1)]+=1
for item in p3[3][0]:
    for opt in item:
        p[int(opt[0]//45+1)]+=1
print(q)
print(m)
print(p)
# print(max2,min2)
# # ff.write("/n")
# # ff.close()
# c1=pywt.waverec2(c2,'db2')
# c1 = np.array(c1,np.uint8)
# c3=np.array(c2[0],np.uint8)
# #cv2.imshow('get',c1)
# #cv2.waitKey()
# #cv2.imshow('getll',c3)
# #cv2.waitKey()
# print(len(c2[1]))
# def idwt2(c2,wavelet,mode,axes):
#     a,ds=c2[0],c2[1:]
#     print(ds)
#
#     d = tuple(np.asarray(coeff) if coeff is not None else None
#                       for coeff in ds[1])
#     d_shapes = (coeff.shape for coeff in d if coeff is not None)
#     print(d_shapes)
#     d_shape=next(d_shapes)
#     idxs = tuple(slice(None, -1 if a_len == d_len + 1 else None)
#                              for a_len, d_len in zip(a.shape, d_shape))
#     a = pywt.idwt2((a[idxs], d), 'db2',mode='symmetric', axes=(-2, -1))
#     return a
# imgs = np.hstack([c22[0],c22[1][0]])
# cv2.imshow('get',imgs)
# cv2.imshow('1',c22[1][1])
# cv2.imshow('2',c22[1][2])
# cv2.waitKey()