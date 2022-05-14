import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Input, AveragePooling2D
import cv2
import os
import numpy as np
import json
import pickle


baseModel = ResNet50(weights="imagenet", include_top=False,
        input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

features = []
for c in range(1, 401):
    print("c:", c)
    video_path = "dataset/0001-0400/clip_{}/clip_{}.mp4".format(c,c)
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    # print("duration:", duration)
    success,image = vidcap.read()
    count = 0

    if not os.path.exists("dataset/0001-0400/clip_{}/images".format(c)):
        os.makedirs("dataset/0001-0400/clip_{}/images".format(c))

    while success:
        cv2.imwrite("dataset/0001-0400/clip_{}/images/frame{}.jpg".format(c, count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1

    # print("total_frames:", frame_count)


    ###################
    # start-end index #
    ###################
    data_dir = "dataset/0001-0400/clip_{}/clip_{}.json".format(c,c)
    with open(data_dir, "r") as json_file:
        st_python = json.load(json_file)
    
    dataset = sorted(st_python['data'].items(), key=lambda x : int(x[0]))
    idxs = []
    
    threshold = 0
    for i, data in enumerate(dataset):
        # print(data[1].keys())
        if i >= threshold:
            for p in ["1", "2"]: 
                try:
                    if "text" in data[1][p].keys():
                        # print("[{}-{}]".format(data[1][p]["text"]["script_start"],data[1][p]["text"]["script_end"]), data[1][p]["text"]["script"], "[person {}]".format(p))
                        threshold = data[1][p]["text"]["script_end"]
                        
                        start = int(data[1][p]["text"]["script_start"])
                        end = int(data[1][p]["text"]["script_end"])
                        idxs.append((start, end))
                except:
                    break

    def convert(path):
        _image = cv2.imread(path)
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
        _image = cv2.resize(_image, (224, 224))

        return _image


    img_dir = "dataset/0001-0400/clip_{}/images".format(c)
    img_data = [[ convert(os.path.join(img_dir, img)) for img in os.listdir(img_dir) if ".jpg" in img and int(img.split(".")[0].split("frame")[-1]) in range(s,e+1)] for s,e in idxs]


    for d in img_data:
        # for im in d:
        #     im = np.array(im)
        #     im = np.expand_dims(im, axis=0)
        d = np.array(d)
        res = model(d)


        res = np.expand_dims(res, axis=0)


        features.append(res)

        

        print(res.shape)


    # for img in os.listdir(img_dir):  
    #     if ".jpg" in img:
    #         i = os.path.join(img_dir, img)

    #         _image = cv2.imread(i)
    #         _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
    #         _image = cv2.resize(_image, (224, 224))

    #         img_data.append(_image)

    # img_data = np.array(img_data) # -> ERROR
    # print("img_data.shape:", img_data.shape)

    # np.save("dataset/0001-0400/clip_{}/images/clip_{}".format(c,c), img_data)

    # img_data = np.load("dataset/0001-0400/clip_{}/images/clip_{}.npy".format(c,c))


# features = np.array(features)
# features = features.reshape(-1, 2, 2048)
# features = [ [features[2*i], features[2*i+1]] for i in range(len(features) // 2) ]
# features = np.squeeze(features, axis=1)

print(len(features))
with open("visual_sample.pickle", "wb") as f:
    pickle.dump(features, f)
# np.save("visual_dataset", features)



# 
# pickle -> class type 고정, list, dict 다 보존
