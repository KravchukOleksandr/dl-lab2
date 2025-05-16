# dl-lab2  
Because the training dataset is only 2.5% of the original [Diabetic Retinopathy 224](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered?utm_source=chatgpt.com), the model (ViT) was not able to learn enough robust features. That's why computer vision feature engineering techniques (LBP, HOG and SIFT extractors) were used to extract some additional features from images.

Results:  
[basic.ipynb](basic.ipynb) - training without feature engineering, Accuracy = 64.63%  
[feature_engineering](feature_engineering.ipynb) - training with feature engineering, Accuracy = 67.57%
