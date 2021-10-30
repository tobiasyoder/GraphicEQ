import matplotlib
import sounddevice as sd
import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy.fft import ifft

from math import*

#from Prototype import DownSampleRate

PLOTTYPE = 'time' #time domain or s domain

#Parameters - The values set to None will get default values from sounddevice library
##These params are needed to open the stream
CHUNKSIZE = 4096
DTYPE = None
NUM_CHANNELS = 1
SAMPLERATE = 44100
BLOCKSIZE = None
LATENCY = None
InputDevice = None
OutputDevice = None


DownSampleRate = 10
ChannelsAry = [1]
Interval = 30
Window = 250

Mapping = [0] #1


ChunkQueue = queue.Queue() #This queue holds chunks

"""
callback gets called for every input block and lets
us specify the output block. This is where we will call all
processing methods
parameters:
indata - This is the input block, stores the audio info as a 2d list. The subarrays have NCHANNELS elements with each one being a channel
    format looks like:
                        [[]
                         []
                         []]

outdata - Anything we put in here will be output to the speaker. Same format 
"""
global I
I = 0
def callback(indata, outdata, frames, time, status):
    global I

    if status:
        print(status)
    

    outdata[:] = ProcessChunk(indata)#ProcessChunk(indata)
    #print(outdata)
    #print(indata)
    #Putting the chunk in the queue so it can be plotted
    ChunkQueue.put(outdata[::DownSampleRate, Mapping])

"""
FT performs the fast fourier transform on the chunk
at the moment it only works for 1 channel chunks in the format:
[[]
 []
 []]
"""
def FT(Chunk):
    Transposed = Chunk.T[0]
    SChunk = fft(np.array(Transposed))
   
   
    OutArray = np.zeros((len(SChunk), 1))
    for i in range(int(len(SChunk)/2)):
        OutArray[i][0] = np.abs(SChunk[i])*.1
    print(OutArray)
    return OutArray

def Inverse_FT(Chunk):
    TimeDomain = 0
    return TimeDomain


def ProcessChunk(InputChunkT):

    Transposed = InputChunkT.T[0]
    InputChunkS = fft(np.array(Transposed))
   
    ProcessChunkS = InputChunkS
    Half = int(len(ProcessChunkS)*(5/10))
    print("half")
    print(Half)
    for i in range(len(ProcessChunkS)):
        if i>Half:
            ProcessChunkS[i] = ProcessChunkS[i]*0
   
    ProcessChunkT = ifft(ProcessChunkS).real
    FormattedChunk = np.zeros((len(ProcessChunkT), 1))
    for i in range(len(ProcessChunkT)):
        FormattedChunk[i][0] = ProcessChunkT[i]
    #print((FormattedChunk))
    return FormattedChunk
  


def update_plot(frame):
    """This is called by matplotlib for each plot update.
    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.
    """
    global plotdata
    if PLOTTYPE == 'time':
        while True:
            try:
                data = ChunkQueue.get_nowait()
            except queue.Empty:
                break
            #The plot needs to be "sliding" from as time passes rather than just updating, so here we are create a shift variable and shifting the old data over
            #so that older data is shifted to the left.
            shift = len(data) 
            plotdata = np.roll(plotdata, -shift, axis=0)
            plotdata[-shift:, :] = data
           

        for column, line in enumerate(lines):

            line.set_ydata(plotdata[:, column])
    else:
        
        PlotDataSize = 0;
        while PlotDataSize<500:
            
            try:
                data = ChunkQueue.get_nowait()
            except queue.Empty:
                #break                         Might cause occasional error?
                a=2
            #The plot needs to be "sliding" from as time passes rather than just updating, so here we are create a shift variable and shifting the old data over
            #so that older data is shifted to the left.
            shift = len(data) 
            plotdata = np.roll(plotdata, -shift, axis=0)
            plotdata[-shift:, :] = data
            PlotDataSize += len(data)
            
        
        
        plotdata = FT(plotdata)
        print("Size of plot data:")
        print(len(plotdata))
        #print(plotdata)
        #plotdata = np.random.rand(441)
        #print(np.shape(plotdata))
        for column, line in enumerate(lines):
            line.set_ydata(plotdata)
    return lines



#Open pass through stream
stream = sd.Stream(device=(InputDevice, OutputDevice),
                   samplerate=SAMPLERATE, blocksize=BLOCKSIZE,
                   dtype=DTYPE, latency=None,
                   channels=1, callback=callback)


#Code to run before the loop actually starts, initializing things

#Stuff for the graphic visualizer:
if PLOTTYPE == 'time':
    length = int(Window * SAMPLERATE / (1000 * DownSampleRate))
else:
   # length = 400
    length = int(Window * SAMPLERATE / (1000 * DownSampleRate))
plotdata = np.zeros((length, NUM_CHANNELS))

fig, ax = plt.subplots()
if(PLOTTYPE=='time'):
    lines = ax.plot(plotdata)
else:
    lines = ax.semilogx(plotdata)
ax.axis((0, len(plotdata), -1, 1))
ani = FuncAnimation(fig, update_plot, interval=Interval, blit=True)


#Inside when the stream is opened, it behaves somewhat like an infinite loop 
#that reapeatadly runs callback
with stream:
    print("TEST")
    I = 1
    if(PLOTTYPE != 'time'):
        plt.xlim([50, 900])
        plt.ylim(-.5, 1.5)
    plt.show()
    input()
    print("cool")
