# DEMO

# NOTES

 The folder called "DEMO" contains my works. However, these projects are heaviry based on tutorails, textbooks and web documentation. Although I typed all of these codes, I did not create them from scrach. I was just following and learning someone else who has better knowledge than me. Therefore I should emphasize these incredble people who put all these resources and education free and available for everybody. I will show the list in the next cell.

 # Resources

 ### [Getting Started with TensorFlow and Deep Learning | SciPy 2018 Tutorial | Josh Gordon ](https://www.youtube.com/watch?v=tYYVSEHq-io&t=5s)

Great intruduction for machine learning using tensorflow explained by developers. This tutorail video containes basic classification, regression, text classification, and overfit/underfit. file "MNIST" is based on this tutorial.

###  [Deep Learning with Python, TensorFlow, and Keras tutorial](https://www.youtube.com/watch?v=wQ8BIBpya2k&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN)

Deep learning tutorial by sendex. He explaines conceput of deep learning and type all codes from scrach to build the entire model. There are tutorial for MNIST, DogvsCat, Bitcoin price. He also provide text based tutorial where you can check all of his codes. Files "MNIST", "DogvsCat", "Crypto" are heaviry based on his tutorial.

###  [Machine Learning with Scikit-learn - Data Analysis with Python and Pandas](https://www.youtube.com/watch?v=BpPJxtOk8uw)

Tutorial for pandas library by sendex. He teaches basic pandas commands and at the end of tutorial, he also shows how to use pandas for machine learning project. File "Diamond" is based on this tutorial.

### [Implement YOLOv3 with OpenCV in Python || YOLOv3 ](https://www.youtube.com/watch?v=R0hipZXJjlI)

Tutorial by Ivan Goncharov to run YOLOv3 object detection with python and opencv. He also shares the resources which helped him to make these codes. 

### [Deep Learning based Object Detection using YOLOv3 with OpenCV ( Python / C++ )](https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/)

The article about how to implement yolo on your computer. This article show python and c++ codes with explanation for each function. 

# Description

### MNIST.ipynb

My first machine learning program. I created this following tensorflow tutorial video at Scipy 2018 by google developers.  I tried to use tensorflow for a while ago but it was complicated (code uses concept of session and graph to perform machine learning) and I could not understand it. However, tensorflow got updates and it became much more user friendly recently. The code has been simplified greatly thanks to kerns implementation. It no longer uses session and graphs but uses layer structure to create neural network model. Simple Dense layer model can be wrote in less than 5 lines of codes which is capable of classifying MNIST datasets. This simple example really helped me to get started with tensorflow and machine learning. I was surprise especially at the fact that you do not need extreme coding knowledge to create simple classifier. I also discovered google Colab which is web-based programming environment where you can learn python programming without worrying about installation of python and various other libraries. I highly recommend google colab especially for beginners since these installation and setting development environment is one of the most difficult part.

### DogvsCat.ipynb

As I learned tensor flow with Scipy 2018 youtube video, I realized Youtube tutorials are one of the most useful resources to learn programming. So I started searching other machine learning tutorials on youtube and I discovered Deep Learning tutorial by sendex where he explained each code and wrote them from scratch so that you can see how these programmers think and write their own program. This DogvsCat.ipynb is based on sendex tutorial. This model is trained by thousand of pictures of dogs and cats provided by Microsoft to classify them. In this tutorial, I learned file structure of these training datasets and main classifying python file. Because this model uses convolutional layers and huge number of pictures, it took a while to train the model with CPU. Thus, I decided to use GPU to train the model. To use GPU, I needed GPU version of tensorflow which requires really complicated installation process involving downloading specific NVIDIA driver and Visual Studio c++ to configure them. However, the training time is reduced dramatically form 12 minute with CPU version to 1 minute with GPU version. I also tested various model with slight changes of number of layers and neurons to search the most efficient model (result DogvsCat/TensorBoard.png).  The results shows the model with 3 convolutional layer and no dense layer has highest 85% accuracy to classify Dog and Cat with 12500 sample pictures for each animals. As a test, I used picture of my dog and the model successfully classified my dog most of the time. It sometime failed to classify the image when animal is covered by object or picture is heavily blurred. After I finished this tutorial, I realized that machine learning involves preprocessing data more than adjusting network model. 

### Diamond.ipynb

I used pandas library to preprocess csv datasets for machine learning. This model predict the price of diamond based on all the other information provided by Kaggle datasets such as size, color, and condition of the diamond. The model is trained by using sklearn library as a regression model. 

### YOLOv3/OD-tutorail.py

After I watched ted talk about object detection (https://www.youtube.com/watch?v=Cgxsv1riJhI), I wanted to run that model on my computer. I discovered tutorial by Ivan Goncharov to run YOLOv3 object detection with python and opencv. I was surprise that yolo could run on my laptop and had decent frame rate (10~15 fps). I downloaded weight and configuration file from official website. I wrote main python program watching Ivan Goncharov tutorial. In this tutorial, I learned how to import already trained model and put detection box on video in realtime. I was surprise at how simple it is to use pre-trained model. Preprocessing video frame and putting detection box were more difficult to implement. 

### Radar.ipynb, Realtime-Detection.py

This radar program is for high frequency radar at laboratory at my university.  Although this model only can distinguish whether there is nothing or metal stick in front of the radar,  this is the first machine learning program which I wrote without following tutorials. Since the radar stores data in csv file, I used pandas library to preprocess datasets. I also used matplotlib library to visualize datasets and implemented live graph to check radar’s wave data in real time (Realtime-Detection.py). In this program, I loaded trained model and preprocessed incoming data in realtime to make prediction in realtime. The model is in progress and I intend to gather more data to implement gesture recognition with the radar. 

