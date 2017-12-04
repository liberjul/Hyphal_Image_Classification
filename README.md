# Hyphal_Image_Classification
Machine learning and image analysis to classify images of fungi taken with brightfield microscopes.

Ths project uses a training set of fungi isolated from soils around the world to train a classifier. The classifier can distinguish between class 1, 
Zygomycetes, and class 2, Ascomycetes and Basiomycetes. Both Support Vector Machine and Random Forest classifiers were used, and the user
can decide which of the two to use for any classification.

To use the repository, ensure that the necessary dependencies are installed, including Numpy, Pickle, Scipy, and Sklearn.

When running the main script, predict_fungus_class.py, the user will be prompted to input only the path to the folder, such as 
"./some_folder/picture.jpeg", and which classifier to use, either SVM or Random Forest. At this time, the performance of the Random
Forest Classifier far exceeds that of the SVM.

I have not tested the application with photos collected from other microscopes, but my setup was 100X magnification, acquired with a
Leica ICC50HD using the LAS EZ software. As more data accumulates, I will continue to update the classifiers and the script.
