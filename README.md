# Audio-Bird-Detection Introduction
Bird audio detection (BAD) is defined as identifying the presence of bird sounds in a given audio recording. In many conventional, remote wildlife-monitoring projects, the monitoring/detection process is not fully automated and requires heavy manual labour to label the obtained data (e.g. by employing video or audio) [1, 2]. In certain cases such as dense forests and low illumination, automated detection of birds in wildlife can be more effective through their sounds compared to visual cues. The problem is challenging as the bird sounds may vary drastically in different species of birds. Automation of this would save lots of manual efforts and makes bioacoustics easier. In a set of audio samples collected in wild, we propose to classify if the audio sample contain bird sounds available in it or not. . This indicates the need for automated BAD systems in various aspects of biological monitoring. For instance, it can be applied in the automatic monitoring of biodiversity, migration patterns, and bird population densities. Using an automated BAD system as pre-processing/filtering step to determine the bird presence would be beneficial especially for remote acoustic monitoring projects, where large amount of audio data is employed. Our work is mainly based on vocal sound made by birds in the audio. Bird calls are often short and serve a particular function such as alarming or keeping the flock in contact. We use spectrograms, which are a visual representation of the magnitude returned by the Short Time Fourier Transform (STFT). STFT is a version of the Discrete Fourier Transform (DFT), which instead of only performing one DFT on a longer signal, splits the signal into partially overlapping chunks and performs the DFT on each using a sliding window. MFCC based features were extracted for each recording for the pre-processing of dataset. Further training and testing of data was performed using SVM. The main contribution being the combined approach of MFCCs and SVMs classification. Such classification technique for the modelling of SVMs to perform audio classification with lower error rates is presented in this paper. Advantages of both resulted in increased accuracy and faster classification method. Basic objective was to decrease error rate with simplified computations.In order to address such problems, we designed a public evaluation campaign focused on a highly general version of the bird audio detection task, intended specifically to encourage detection methods which are able to generalise well  to species. In this work, we present the new acoustic datasets which we collated and annotated, the design of the challenge, and its outcomes, with new machine learning methods able to achieve strong results despite the difficult task. We analyse the submitted system outputs for their detection ability as well as their robust calibration; we perform a detailed error analysis to inspect the sound types that remain difficult for machine learning detectors, and apply the leading system to a separate held‐out dataset of night flight calls. We conclude by discussing the new state of the art represented by the machine learning methods that excelled in our challenge, the quality of their outputs, and the feasibility of deployment in remote monitoring projects.

#  MOTIVATION & PROBLEM STATEMENT
 Automatic identification of bird sound in the audio clip helps researcher to find out their : ○ Migration patterns ○ Automatic Wildlife   Monitoring ○ Estimation of bird species richness and abundance .
 
 ● Using machine learning we would like to train the different models that serve as Automatic Bird Detection (BAD) system using available dataset 

● In a set of audio samples collected in wild, we propose to classify if the audio sample contain bird sounds available in it or not. 

● Different bird produces different types of sounds which makes it challenging from the sound whether it was actually a bird or not.

# DATASET DESCRIPTION

● Available dataset i.e. field recording dataset (Freefield) used.

● Freefield has recordings containing over 7000 samples. 

● These datasets are comprised of 10-second 16-bit 44.1 kHz audio recordings that were manually labeled with binary labels. 

● Whole data set is divided in training and testing in ratio of 80:20.

● Performed data augmentation through time shifting in samples to overcome the imbalance in data. 

# PREPROCESSING ALGORITHM
Step 1: Data Augmentation performed to balance +ve and -ve data samples. Step 2: After this, we extract MFCC coefficients by applying following five steps:
● Pre- emphasis filter in spatial domain.

● Split the signal into frames and apply suitable window function.

● Take the N-point FFT of these windowed frame signals (STFT).

● Apply Mel-scale filter banks (Triangular Filter bank) to the amplitude spectrum of  STFT of signals. 

● Apply Discrete Cosine Transform (DCT) for decorrelation of the filter banks.
Step 3: MFCC feature of size 1000x64 are fed to the pre trained VGGish model. VGGish feature extractor, provided by Google Audioset team used.            (4 layers of convolutional followed by max pooling, 3 fully connected layer). Final feature vector of size 11x128 obtained. Step 4: Baseline and various ML models applied on the extracted feature vectors (Flatten for SVM and MLP).
Step 1: Data Augmentation performed to balance +ve and -ve data samples. Step 2: After this, we extract MFCC coefficients by applying following five steps: 
● Pre- emphasis filter in spatial domain. 

● Split the signal into frames and apply suitable window function. 

● Take the N-point FFT of these windowed frame signals (STFT).

● Apply Mel-scale filter banks (Triangular Filter bank) to the amplitude spectrum of  STFT of signals. 

● Apply Discrete Cosine Transform (DCT) for decorrelation of the filter banks.

Step 3: MFCC feature of size 1000x64 are fed to the pre trained VGGish model. VGGish feature extractor, provided by Google Audioset team used.            (4 layers of convolutional followed bye. max pooling, 3 fully connected layer). Final feature vector of size 11x128 obtained. Step 4: Baseline and various ML models applied on the extracted feature vectors (Flatten for SVM and MLP).

![image](https://user-images.githubusercontent.com/54641886/78532728-5ede0680-7805-11ea-8186-35bc26cad4c6.png)

# RESULTS
![image](https://user-images.githubusercontent.com/54641886/78533189-30146000-7806-11ea-9e63-0c026cfe7f47.png)




