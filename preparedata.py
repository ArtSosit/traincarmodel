import cv2
import requests
import numpy as np
import pickle
import os 
import base64 

url = 'http://localhost:8080/api/genhog'

# function ใช้ในการรับภาพแล้วส่งไปยัง API ที่รันอยู่บน docker
def img2vec(img):
    v, buffer = cv2.imencode(".jpg", img) 
    img_str = base64.b64encode(buffer) 
    data = "image data,"+str.split(str(img_str),"'")[1] 
    response = requests.post(url, json={"image_base64":data})
    
    return response.json()
# โหลดรูปภาพ0จาก Folder Train เพื่อเเปลงภาพให้กลายมาเป็น feature vector ด้วยการเรียก API ที่รันอยู่บน docker desktop 
PathTrain = 'train'
FeatureVectorTrain = []

for y in os.listdir(PathTrain): # วน loop เพื่อเเยก folder ย่อย,yจะเก็บ folderยี่ห้อรถ
    for fn in os.listdir(os.path.join(PathTrain,y)): # ทำการวน loop เพื่อเเยกไฟล์ที่อยู่ใน y โดยผลลัพธ์จะได้ชื่อไฟล์ออกมาเช่น 90.jpg
        img_file_name = os.path.join(PathTrain,y)+"/"+fn # สร้างชื่อไฟล์ภาพ output = train\Audi/90.jpg
        X = cv2.imread(img_file_name) 
        res = img2vec(X) # ทำการส่ง X เข้าไปเพื่อ requests ไปยัง apiเพื่อหาเอกลักษณ์ของภาพ
        vec = list(res["vector"])
        vec.append(y)
        FeatureVectorTrain.append(vec)



PathTest = 'test'
FeatureVectorTest = []

for y in os.listdir(PathTest): 
    for fn in os.listdir(os.path.join(PathTest,y)):
        img_file_name = os.path.join(PathTest,y)+"/"+fn 
        X = cv2.imread(img_file_name) 
        res = img2vec(X)
        vec = list(res["vector"])
        vec.append(y)
        FeatureVectorTest.append(vec)


for index, data in enumerate(FeatureVectorTest):
    print(data)
    if index == 1:
        break

# ส่เขียนข้อมูลจาก FeatureVectorTrain and FeatureVectorTest ลงในไฟล์
write_path = "featurevectortrain.pkl"
pickle.dump(FeatureVectorTrain, open(write_path,"wb"))
print("data preparation is done")

write_path = "featurevectortest.pkl"
pickle.dump(FeatureVectorTest, open(write_path,"wb"))
print("data preparation is done")