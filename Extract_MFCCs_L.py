import os
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import pandas as pd
import scipy
from pathlib import Path
import glob2
import IPython.display as ipd

def read_audio_files_in_directory(directory_path):
    # Initialize a dictionary to store the audio data
    audio_data_dict = {}

    # List all files in the directory
    all_files = os.listdir(directory_path)

    # Filter for .wav files
    wav_files = [file for file in all_files if file.endswith('.wav')]
    
    # Iterate over each WAV file
    for wav_file in wav_files:
        try:
            # Combine directory path and file name to get the full file path
            file_path = os.path.join(directory_path, wav_file)
            
            # Read the WAV file into an audio segment
            audio_data = AudioSegment.from_wav(file_path)
            # Store the audio data in the dictionary with the file name as the key
            audio_data_dict[wav_file] = audio_data
        
        except Exception as e:
            
            # If an error occurs, print it out
            print(f"Error reading {wav_file}: {e}")
            

    return audio_data_dict

# Provide the path to your  folder
dir_path = 'Datasets_32'
audio_files_data = read_audio_files_in_directory(dir_path)



# Print the names of the files
FileNmae = []
for file_name in audio_files_data.keys():
    FileNmae.append(file_name)
    print(file_name)
    
    
FileNmae[0]    

def Read_wav_Files(Name):
    # Read WAV file using soundfile library, getting raw data and sample rate
     r2, fs2 = sf.read(Name, dtype='float32')
     
     print(len(r2))

     print(fs2)
     
     return r2, fs2

 


# Conversion functions
def Conversion_hz_to_mel(hz):
    # Convert frequency from Hz to Mel scale
    return 1125 * np.log(1 + hz / 700.0)

def mel_to_hz(mel):
    # Convert frequency from Mel scale to Hz

    return 700 * (np.exp(mel / 1125) - 1)

# Designing the Mel filterbank
def mel_filterbank(num_filters, fft_size, fs2, min_hz, max_hz):
    # Convert frequency to Mel scale
    min_mel = Conversion_hz_to_mel(min_hz)
    max_mel = Conversion_hz_to_mel(max_hz)
    
    # Create linearly spaced points between the minimum and maximum Mel values
    mel_points = np.linspace(min_mel, max_mel, num_filters + 2)
    # Convert these points back to Hz scale
    hz_points = mel_to_hz(mel_points)
    
    # Convert Hz to corresponding FFT bin numbers
    bin_points = np.floor((fft_size) * hz_points / fs2).astype(int)

    filters = np.zeros((num_filters, fft_size // 2 ))
    
    # Construct the triangular filters
    for i in range(1, num_filters + 1):
        for j in range(fft_size // 2 ):
            if j < bin_points[i-1]:
                filters[i-1, j] = 0
            elif j < bin_points[i]:
                filters[i-1, j] = (j - bin_points[i-1]) / (bin_points[i] - bin_points[i-1])
            elif j < bin_points[i+1]:
                filters[i-1, j] = 1 - (j - bin_points[i]) / (bin_points[i+1] - bin_points[i])
            else:
                filters[i-1, j] = 0
    
    return filters



fft_size = 512
min_hz = 10


def linearRectangularFilterbank(magspec, numChannels):
    
    
    # Initialize the filterbank output vector
    #fbank = np.zeros(num_channels)
    
    # Replace this with your 256-point magnitude spectrum
   
    #print(magspec)
    
    # Determine the width of each channel in the filterbank
    channel = len(magspec) // numChannels
   
    # Initialize the filterbank output
    fbank = np.zeros(numChannels)
    for i in range(numChannels-1):
        # Sum the magnitude spectrum over the channel width to get filterbank energy
        fbank[i] = sum(magspec[i*channel:(i+1)*channel])
    #print(fbank)
    
    
    return fbank


def magphasE(frame):
    frameLength = len(frame)
    
    # Apply a Hamming window to the frame
    h = np.hamming(frameLength)
    hammedWindow = frame * h
    
    # Perform the FFT on the windowed frame
    fft_frame=np.fft.fft(hammedWindow)
    
    # Calculate the magnitude spectrum
    magSpec = np.abs(fft_frame)
    
    # Only take the first half of the spectrum (symmetric for real-valued signals)
    magSpec = magSpec[:frameLength//2]
    
    # Calculate the phase spectrum
    phaseSpec = np.angle(fft_frame)
    
    
    return magSpec, phaseSpec

# magSpec , phase = magphasE(r2)
# plt.figure()
# plt.plot(magSpec)
# plt.figure()  
#plt.plot(phase)

frameLength = 512
# Extract frames from an audio file:#
#Choose  a frame l;ength of 20/30 ms which is the equiv of 320 samples, 512 samples
def utterance(r2, frameLength, numChannels,Name):
    # Calculate the total length of the audio and the number of frames
    fileLength = len(r2)
    numFrames = fileLength / frameLength
    
    
    # Initialize lists to hold magnitude spectra, phase spectra, and features
    Maglist = []
    phaselist = []
    #features = []
    Features = []

        # Process each frame of the audio
    for i in range(0,fileLength-frameLength,frameLength//2):
        start = i
        end = i+frameLength
        
        #print(start)
        #print(end)
        frame = r2[start:end]
        
        magS, phase = magphasE(frame)
        #print(magS.shape)
        #feature = linearRectangularFilterbank(magS, numChannels) 
        
        Filterss = mel_filterbank(numChannels, fft_size, fs2, min_hz, fs2 / 2)
        # print(Filterss.shape)
        melfilterbank = np.matmul(Filterss,magS)
        
        Log = np.log(melfilterbank)
        #scipy.fft.dct(Log)
        Dct = scipy.fft.dct(Log)
        
        Features.append(Dct)
       
        #features.append(feature)
        Maglist.append(magS)
        phaselist.append(phase)
    
    
    np.save(Name,Features)
    
    return  np.array(Maglist), np.array(phaselist) , frameLength , numFrames, np.array(Features), numChannels
        

LIst = []
for i in range(len(FileNmae)):
    Name = FileNmae[i]
    r2,fs2 =Read_wav_Files(Name)
    
    magSpec , phase = magphasE(r2)
    
    Mag , phase,Framelength, NumFrame,Features,numChannels = utterance(r2, frameLength, 32 , Name)
    # Visualization
    print(magSpec)
    LIst.append(Features)
    print(len(r2))
    plt.figure()
    plt.plot(magSpec)
    filters = mel_filterbank(numChannels, fft_size, fs2, min_hz, fs2 / 2)
     
    plt.title('Mel Filterbank')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()
    for i in range(numChannels):
        plt.plot(filters[i, :])
        





# audio_files_data.keys