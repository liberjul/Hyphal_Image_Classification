import numpy as np
import pickle
from scipy.ndimage import imread
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


def photo_transform(im_name):
    '''
    Reads the image file and applies fast Fourier transform, then flattens it for use by the PCA and SVM algorithms.
    Input: image name, such as "./Fungi Pics/JU-15B.tif".
    Output: flattened and 2D array of frequency space
    '''
    im = imread(im_name, flatten=True) # read image and converts it to gray
    Fs = np.fft.fft2(im) # run fast Fourier transform on gray image
    F2 = np.fft.fftshift(Fs) # move the zero frequency component to the center
    psd2D = np.abs(F2) # remove imaginary values
    flat_psd = psd2D.flatten() # flatten the array
    flat_psd = flat_psd.reshape((1, -1)) # make it 2D for PCA and SVM
    return flat_psd
def photo_pred(flat_psd, classifier):
    '''
    Makes a prediction of which class the fungi is in when given a transformed image.
    Input: flattened 2D fft image from photo_transform
    Output: class prediction
    '''
    if classifier == 'SVM':
        pca = pickle.load(open("hyphal_image_pca.obj", 'rb')) # load up-to-data PCA and SVM classifier
        clf = pickle.load(open("hyphal_image_classifier.obj", 'rb'))
        pca_psd = pca.transform(flat_psd) # transform with the PCA
        pred_label = clf.predict(pca_psd) # predict with the SVM
    else:
        clf2 = pickle.load(open("hyphal_image_RF_classifier.obj", 'rb')) # Load up-to-date Random Forest classifier
        pred_label = clf2.predict(flat_psd) # predict with the Random Forest Classifier
    if pred_label == 1: # return prediction decision
        return "Zygomycete"
    if pred_label == 2:
        return "Ascomycete or Basidiomycete"
def main(im_name, classifier):
    flat_psd = photo_transform(im_name)
    pred_label = photo_pred(flat_psd, classifier)
    print(pred_label)
if __name__ == "__main__":
    im_name = str(input("Path name of photo:"))
    classifier = str(input("Which classifier: [SVM]/[RF]?"))
    main(im_name, classifier)
