# Real-Time-Face-Recognition

## Objective:
The objective of the proposed solution is to recognize a face of a person from a trained data set. The solution recognizes the face and displays the name of the person.

The solution is fast and takes less execution time and has a great accuracy even for a smaller number of images in the training data set. It provides a hassle-free solution to an entry and exit management system based on facial recognition in different types of communities.

The solution determines the number of times a person has entered and exited through the system where the solution is implemented.

## Brief Description about the Concept:
There are three easy steps to computer coding facial recognition, which are similar to the steps that the brain uses for recognizing faces. These steps are:

Data Gathering: Gather faces data of training space.

Train the Recognizer: Train the model with extracted data.

Recognition:The trained model predicts labels.

OpenCV recognizer framework used is Local Binary Patterns Histograms (LBPH) – cv2.face.LBPHFaceRecognizer_create()

The LBPH Face Recognizer Process It takes a 3×3 window and move it across one image. At each move (each local part of the picture), compare the pixel at the center, with its surrounding pixels. Denote the neighbors with intensity value less than or equal to the center pixel by 1 and the rest by 0. After consuming the whole image data, list of local binary patterns is created. The histogram obtained is unique for each label. The algorithm also keeps track of which histogram belongs to which person. Later during recognition, the process is as follows:

* Feed a new image to the recognizer for face recognition.

* The recognizer generates a histogram for that new picture.

* It then compares that histogram with the histograms it already has. 4. Finally, it finds the best match and returns the person label associated with that best match.

Coding Face Recognition using Python and OpenCV The Face Recognition process is divided into three steps:

* Prepare Training Data: Read train data and assign an integer label to each image data.

* Train Face Recognizer: Train OpenCV's LBPH recognizer by feeding it the data we prepared in step 1.

* Prediction: Introduce some test images to face recognizer and see if it predicts them correctly.

## Run the program like this:

*python face recg.py*

Update: Now supports OpenCV3.

To run the OpenCV3 version, run python webcam_cv3.py haarcascade_frontalface_default.xml
