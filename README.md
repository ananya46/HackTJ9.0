# Pneumonia Detector

The Pneumonia Detector is a tool made to detect the infectious disease pneumonia based off of a patient's lung x-ray scan. The detector is a CNN model that was trained off of a dataset from kaggle: https://www.kaggle.com/pranavraikokte/pneumoniadataset. The highest accuracy we reached was 80%, but this value may vary.


# Instructions

1. Download HackTJ.ipynb and launch your computer's Terminal (this may also be called 'cmd' for Windows users)
2. Install the following packages using ```pip install```: keras, tensorflow, numpy, matplotlib, sklearn, and imutils.
3. To test the model, upload your new lung x-ray image into the directory and enter into cell:
      ```targetimage=cv2.imread(YOUR_FILENAME_HERE.jpg')```
      ```targetimage=cv2.resize(targetimage,(256,256))```
4. The data will be returned in the format [x, y]. If the x value is greater than the y valuem, pneumonia is not present in the x-ray.

