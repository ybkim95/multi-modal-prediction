# Multimodal Approach to Next-Utterance Timing Prediction

Reproducing IVA '21 [paper](https://dl.acm.org/doi/abs/10.1145/3472306.3478360) "Multimodal and Multitask Approach to Listener's Backchannel Prediction: Can Prediction of Turn-changing and Turn-management Willingness Improve Backchannel Modeling?".

![image](https://user-images.githubusercontent.com/45308022/168421485-2da51c95-0931-416f-8cf2-1be07a29543c.png)
<br><br>


## Requirements
1. python==3.9
2. torch==1.8.0
3. sentence-transformers==2.2.0
4. download the [data](https://dl.acm.org/doi/abs/10.1145/3472306.3478360) from AI Hub 
5. [vggish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
<br>


## How to run
1. Run ```python3 linguistic_modality.py ``` to generate vectorized embedding data into .pickle format.
2. Run ```python3 visual_modality.py ``` to generate image feature vectors into .pickle format.
3. Inside .audio_features/, run ```python3 audio_modality.py ``` to generate audio feature vectors into .pickle format.
4. Run ```python3 main.py``` to train the model. 
<br>


## Difference compared to the paper
1. Instead of using Japanese MM-TMW Corpus mentioned in paper, I have used [multi-modal data](https://dl.acm.org/doi/abs/10.1145/3472306.3478360) from AI Hub. 
2. Due to the difference in language type, I used different BERT embedding model to process the language input. 
3. Focused on implementing the turn-chaning prediction rather than willingness and backchannel prediction. 
<br>


## Future Works
1. Combine the work from [2],[3] in order to output controlled-sentence with at a given timing.
2. Predict the participant's affection from the multi-modal data (e.g. visual, language, audio)
<br>


## Reference
1. Ishii, R., Ren, X., Muszynski, M., & Morency, L. P. (2021, September). Multimodal and Multitask Approach to Listener's Backchannel Prediction: Can Prediction of Turn-changing and Turn-management Willingness Improve Backchannel Modeling?. In Proceedings of the 21st ACM International Conference on Intelligent Virtual Agents (pp. 131-138).
2. Hu, Z., Yang, Z., Liang, X., Salakhutdinov, R., & Xing, E. P. (2017, July). Toward controlled generation of text. In International conference on machine learning (pp. 1587-1596). PMLR.
3. Dathathri, S., Madotto, A., Lan, J., Hung, J., Frank, E., Molino, P., ... & Liu, R. (2019). Plug and play language models: A simple approach to controlled text generation. arXiv preprint arXiv:1912.02164.
