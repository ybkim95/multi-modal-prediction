import json
import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer


# Embedding
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

embeddings = []
for c in range(1, 401):
    data_dir = "dataset/0001-0400/clip_{}/clip_{}.json".format(c,c)
    print("c:", c)

    # print("\nProcessing {} ...\n".format(data_dir))

    with open(data_dir, "r") as json_file:
        st_python = json.load(json_file)

    # print(type(st_python))
    #print(st_python.keys())
    #print(st_python['situation'])
    #print(st_python['video_size'])
    #print(st_python['clip_id'])
    #print(st_python['category'])

    dataset = sorted(st_python['data'].items(), key=lambda x : int(x[0]))

    # print("Number of frames: {}".format(st_python['nr_frame']))
    # print()

    threshold = 0
    for i, data in enumerate(dataset):
        # print(data[1].keys())
        if i >= threshold:
            for p in ["1", "2"]: 
                try:
                    if "text" in data[1][p].keys():
                        # print("[{}-{}]".format(data[1][p]["text"]["script_start"],data[1][p]["text"]["script_end"]), data[1][p]["text"]["script"], "[person {}]".format(p))
                        threshold = data[1][p]["text"]["script_end"]

                        sentence = data[1][p]["text"]["script"]
                        print(sentence)

                        embedding = model.encode(sentence)
                        embedding = np.expand_dims(embedding, axis=0)
                        
                        embeddings.append(embedding)
                        

                except:
                    break
    
    
    # print(embeddings.shape)   # (N x 768)

# embeddings = np.array(embeddings)
# embeddings = embeddings.reshape(-1, 2, 768)
# embeddings = np.expand_dims(embeddings, axis=0)
# print(embeddings.shape)

# embeddings = [ [embeddings[2*i], embeddings[2*i+1]] for i in range(len(embeddings) // 2) ]


# np.save('linguistic_sample', embeddings)

with open("linguistic_sample.pickle", "wb") as f:
    pickle.dump(embeddings, f)