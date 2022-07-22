## JBcnConf 2022 slides and code for the talk:  
## **[Gentle introduction to machine learning algorithms, with examples](https://www.jbcnconf.com/2022/infoTalk.html?id=6272e73371e11d0858e82245)**

Here are the contents of this repo:
- The slides are in OpenDocument format inside folder slides
- Admission_Predict.csv is the [UCLA Admissions dataset](https://www.kaggle.com/code/ashishpatel26/everything-about-ucla-admission-criteria/data) downloaded from Kaggle.
- PLS_PCA_Kmeans.ipynb is a Jupyter notebook that shows how to use Partial Least Squares (PLS), Principal Component Analysis (PCA) and K-means clustering with the dataset.
- paddleocr.ipynb is a Jupyter notebook that runs Spanish car's license plate object detection and license plate recognition from the images inside folder "real_imgs".
- wpod-net.h5 and wpod-net.json is the PaddleOCR model and weights
- an artificial dataset of license plates in a zip file inside "/test" folder. You have to uncompress it to use paddleocr.ipynb

The rest of files are only needed if you want to create an artificial spanish car license plate dataset instead of the images in "/test" folder.
If that's the case, I suggest you to read the instructions in Matthew Earl's [deep-anpr GitHub repo](https://github.com/matthewearl/deep-anpr) but
use the common.py and gen.py Python files from this repository (Mathew's are for UK license plates). 

Also, I found that spanish license plates use 
the following font type which some blog post (which I can't find now) mentioned was used in spanish cars license plates, and
that's the one you should use instead of the one mentioned in the original repo.
 
The general idea is that a license plate number is randomly generated out of the characters and numbers included in common.py,
following the pattern 4 numbers + 3 letters (this is defined in gen.py function "generate_code()"). Then, it is 
transformed (scaled and rotated randomly) and embedded in a random crop of a randomly selected background image.

I didn't include in this repository neither the original [SUN images dataset](http://groups.csail.mit.edu/vision/SUN/) (a 36Gb long dataset) out of which extractbgs will sample
3Gb for the background images, neither the 3Gb sample of background images, I just provide the final artificial dataset in "/test".

The script ocr_prediction.py can be used to assess the quality of PaddleOCR predictions on all the images from test folder 
(it will take some time, there are 10000 images in my test folder).