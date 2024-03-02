import pywt
import cv2
import numpy as np
from nltk.metrics.distance import edit_distance
def imgset(img,wa_img,hid_img):
    waterimg=cv2.imread(wa_img)
    img=cv2.imread(img)
    CA,(CB,CC,CD)=pywt.wavedec2(img,'db2',level=1)
    cA, (cB, cC, cD) = pywt.wavedec2(waterimg, 'db2', level=1)
    CA=CA+cA*0.1
    CB = CB + cB * 0.1
    CC = CC + cC * 0.1
    CD = CD + cD * 0.1
    newimg=pywt.waverec2([CA,(CB,CC,CD)],'db2')
    #cv2.imshow(newimg)
    cv2.imwrite(hid_img,newimg)
#imgset('./3.jpg','./4.jpg','./10.jpg')
def imgget(img,hid_img,outwater_img):
    waterimg=cv2.imread(hid_img)
    img=cv2.imread(img)
    CA,(CB,CC,CD)=pywt.wavedec2(img,'db2',level=1)
    cA, (cB, cC, cD) = pywt.wavedec2(waterimg, 'db2', level=1)
    CA=(cA-CA)*10
    CB=(cB-CB)*10
    CC = (cC - CC) * 10
    CD = (cD - CD) * 10
    newimg=pywt.waverec2([CA,(CB,CC,CD)],'db2')
    #cv2.imshow(newimg)
    cv2.imwrite(newimg,outwater_img)
#imgset('./3.jpg','./10.jpg','./11.jpg')

def converse():
    waterimg = cv2.imread('./001.png')
    waterimg2 = cv2.imread('./002.png')
    p1 = pywt.wavedec2(waterimg2, "bior1.3", level=1)
    p2 = pywt.wavedec2(waterimg, 'bior1.3', level=1)
    sum1=0
    m=0
    print(np.array(p1[1]).shape)
    qt=[]
    print(p1[1][0][0][0] , p2[1][0][0][0])
    for i in range(len(p1[1])):
        for j in range(len(p1[1][0])):
            for kk in range(len(p1[1][0][1])):
                m+=1
                try:
                    q1 = p1[1][i][j][kk] - p2[1][i][j][kk]
                    if abs(q1[0]) + abs(q1[1]) + abs(q1[2]) > 10:
                        # print(q1,p1[1][i][j][kk])
                        sum1 += 1
                        qt.append(q1)
                except:
                    break

    print(sum1,m)
#converse()
def edit_distan(seq1, seq2):
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)

    # 初始化一维数组来存储编辑距离的中间结果，大小为len_seq2+1
    dp = [0] * (len_seq2 + 1)

    for j in range(len_seq2 + 1):
        dp[j] = j

    # 动态规划计算编辑距离
    for i in range(1, len_seq1 + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len_seq2 + 1):
            temp = dp[j]
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = min(prev, dp[j - 1], dp[j]) + 1
            prev = temp

    return dp[len_seq2]
def compute():
    waterimg = cv2.imread('./t5.jpg')
    waterimg2 = cv2.imread('./t6.jpg')
    p1 = pywt.wavedec2(waterimg2, "bior1.3", level=1)
    p2 = pywt.wavedec2(waterimg, 'bior1.3', level=1)
    qt=[]
    qt2=[]
    q = 0 if (p1[0][0][0][0] - p1[0][0][1][0]) > 0 else 1
    print(q)
    for i in range(len(p1[0])):
        for j in range(len(p1[0][0])-1):
            q1 = 0 if p1[0][i][j][0] - p1[0][i][j+1][0]<0 else 1
            qt.append(q1)
    for i in range(len(p2[0])):
        for j in range(len(p2[0][0])-1):
            q2 = 0 if p2[0][i][j][0] - p2[0][i][j + 1][0] < 0 else 1
            qt2.append(q2)
    print(len(qt))
    print(len(qt2))
    edit_dis=edit_distan(str(qt),str(qt2))
    print(edit_dis/len(qt))
compute()