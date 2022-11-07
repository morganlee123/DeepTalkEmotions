
#from types import NoneType
from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from encoder import audio
from vocoder import inference as vocoder
import numpy as np
import torch
import librosa
from utils.sigproc import *
import torchvision.transforms as transforms
from pathlib import Path
import demo_config as config
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

def fCNN_encoder(file_path, model_save_path, sampling_rate=8000, n_channels=1, duration = None, is_cmvn = False, normalize=True,):
    # Load model model_save_path
    from encoder.models import OneD_Triplet_fCNN as network
    model = network.cnn()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load audio from file_path
    win = np.hamming(int(sampling_rate*0.02))
    frame = get_frame_from_file(file_path, win=win, sr=sampling_rate, n_channels=n_channels, duration = duration, is_cmvn=is_cmvn)
    data = np.expand_dims(frame, axis=2)
    transform =transforms.Compose([transforms.ToTensor()])
    data = transform(data)
    data = data.unsqueeze(0)
    data = data.float()

    ## Evaluate the audio using the model
    x1, _ = model(data)
    x1_d = x1.data.cpu().float().numpy().flatten()
    embed_input = np.concatenate((x1_d, x1_d), axis=0)

    if(normalize):
        embed = embed_input / np.linalg.norm(embed_input)

    return embed





def OneD_Triplet_CNN_encoder(file_path, model_save_path, ftr_type = 'MFCC-LPC', sampling_rate=16000, n_channels=1, duration = 2.01, normalize=True,):
    # Load model model_save_path
    from encoder.models import OneD_Triplet_fCNN as network
    model = network.cnn()
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load audio from file_path
    win = np.hamming(int(sampling_rate*0.02))
    inc = int(win.shape[0]/2)
    input_audio, sr = librosa.load(file_path, sr=sampling_rate)
    order = 20
    preemphasis = True
    includeDerivatives = True
    if ftr_type == 'MFCC-LPC':
        frame = get_mfcc_lpc_feature(input_audio, sampling_rate, order = order, preemphasis = preemphasis, includeDerivatives = includeDerivatives, win = win, inc = inc)
    data = frame
    transform =transforms.Compose([transforms.ToTensor()])
    data = transform(data)
    data = data.unsqueeze(0)
    data = data.float()

    ## Evaluate the audio using the model
    x1 = model(data)
    x1_d = x1.data.cpu().float().numpy().flatten()
    embed_input = np.concatenate((x1_d, x1_d), axis=0)
    if(normalize):
        embed = embed_input / np.linalg.norm(embed_input)

    return embed


def DeepTalk_encoder(file_path, model_save_path, module_name, preprocess=True, normalize=True, sampling_rate=16000, duration=None):

    encoder.load_model(model_save_path, module_name=module_name)

    if(preprocess):
        wav = Synthesizer.load_preprocess_wav(file_path)
        ref_audio = encoder.preprocess_wav(wav)
    else:
        ref_audio, sr = librosa.load(file_path, sr=sampling_rate)

    if(duration is not None):
        ref_audio = ref_audio[0:int(duration*sampling_rate)]

    embed, partial_embeds, _  = encoder.embed_utterance(ref_audio, using_partials=True, return_partials=True)

    if(normalize):
        embed = embed / np.linalg.norm(embed)

    return embed


def DeepTalk_synthesizer(encoder_embedding, output_text, model_save_path, low_mem = False):
    synthesizer = Synthesizer(model_save_path, low_mem=low_mem)
    texts = output_text
    texts = texts.split("\n")
    embeds = np.stack([encoder_embedding] * len(texts))
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)
    mel = spec

    return mel, breaks

def DeepTalk_vocoder(synthesized_mel, breaks, model_save_path, normalize=True):
    vocoder.load_model(model_save_path)
    no_action = lambda *args: None
    wav1 = vocoder.infer_waveform(synthesized_mel, progress_callback=no_action, normalize=normalize)

    # Add breaks
    b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [wav1[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
    wav1 = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])
    wav1 = wav1 / np.abs(wav1).max() * 0.97
    return wav1

def split(wav_file_path):
    from pydub import AudioSegment
    from pydub.utils import make_chunks

    myaudio = AudioSegment.from_file(wav_file_path , "wav") 
    chunk_length_ms = 1000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

    #Export all of the individual chunks as wav files
    import glob
    import os
    for i in glob.glob('chunks/*'):
        os.remove(i)

    chunks_paths = []
    for i, chunk in enumerate(chunks):
        chunk_name = "chunks/chunk{0}.wav".format(i)
        print("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")
        chunks_paths.append(chunk_name)
    return chunks_paths

# Run with deeptalk env
def run_DeepTalk_demo(ref_audio_path='samples/ref_VCTKp240.wav', output_text='Hello World',
enc_model_fpath=config.enc_model_fpath, enc_module_name=config.enc_module_name,
syn_model_dir=config.syn_model_dir, voc_model_fpath=config.voc_model_fpath, key_embed=None):
    class hyperparameter:
        def __init__(self):

            self.enc_model_fpath = enc_model_fpath
            self.enc_module_name = enc_module_name
            self.syn_model_dir = syn_model_dir
            self.voc_model_fpath = voc_model_fpath

            self.enc_normalize = False
            self.voc_normalize = True
            self.low_mem = False        # "If True, the memory used by the synthesizer will be freed after each use. Adds large "
                                        # "overhead but allows to save some GPU memory for lower-end GPUs."
            self.no_sound = False       # If True, audio won't be played.
            self.sampling_rate = 16000  ## 16000: For mel-spectrogram based methods; 8000: For fCNN base methods
            self.ref_audio_path = ref_audio_path
            self.output_text = output_text

    args = hyperparameter()
    

    ## Load trained models: Encoder, Synthesizer, and Vocoder
    # this line sets to use the other gpu cuz i filled the other with memory
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    encoder.load_model(args.enc_model_fpath, module_name=args.enc_module_name)
    synthesizer = Synthesizer(args.syn_model_dir, low_mem=args.low_mem)
    vocoder.load_model(args.voc_model_fpath)

    ## Encoding stage
    print('---------------------------------------------------------------')
    print('Stage 1/3: Encoder')
    print('---------------------------------------------------------------')

    # Split the file into 1 second segments
    #chunk_paths = split(args.ref_audio_path)

    total_embeds = []
    #for path in chunk_paths:
    wav = Synthesizer.load_preprocess_wav(args.ref_audio_path)#path)
    ref_audio = encoder.preprocess_wav(wav)
    
    import librosa
    import numpy as np

    #frame_len, hop_len = 2048, 512
    frame_len, hop_len = 22000, 220 # 1 second frame (assuming frame loss from 24000) with hop of 10 ms
    print(librosa.get_duration(ref_audio))
    print(len(ref_audio))    
    frames = librosa.util.frame(ref_audio, frame_length=frame_len, hop_length=hop_len,axis=0)
    #windowed_frames = np.hanning(frame_len).reshape(-1, 1)*frames
    #print(len(ref_audio))
    #print(len(frames))

    for f in frames:
        #print(f)
        embed, partial_embeds, _  = encoder.embed_utterance(f, using_partials=True, return_partials=True, key_embed = key_embed)
        if(args.enc_normalize): 
            embed = embed / np.linalg.norm(embed)

        try:
            if(embed.shape[0]==128):
                embed = np.concatenate((embed, embed), axis=0)
        except AttributeError:
            continue

        total_embeds.append(embed)

    # Return the array of embeddings per the splits
    print('Total embeddings returned:', len(total_embeds) )
    return total_embeds
    
#returned_embeddings = run_DeepTalk_demo(ref_audio_path='/scratch/sandle20/SpeechEmotionRecognitionExperiments/Samples/1043_ITS_SAD_XX.wav')

#print(len(returned_embeddings))

"""
    ## Synthesizing stage
    print('---------------------------------------------------------------')
    print('Stage 2/3: Synthesizer')
    print('---------------------------------------------------------------')
    texts = args.output_text
    # texts = re.split(',|.',texts)
    texts = re.split(r'[,.]\s*', texts)
    texts[:] = [x for x in texts if x]
    print(texts)
    # texts = texts.split("\n")
    # texts = texts.split(".")
    # texts = texts.split(",")
    embeds = np.stack([embed] * len(texts))
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    breaks = [spec.shape[1] for spec in specs]
    synthesized_mel = np.concatenate(specs, axis=1)

    ## Vocoding stage
    print('---------------------------------------------------------------')
    print('Stage 3/3: Vocoder')
    print('---------------------------------------------------------------')
    no_action = lambda *args: None
    wav1 = vocoder.infer_waveform(synthesized_mel, progress_callback=no_action, normalize=args.voc_normalize)
    # Add breaks
    b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [wav1[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
    wav1 = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])
    synthesized_wav = wav1 / np.abs(wav1).max() * 0.97

    return synthesized_wav, Synthesizer.sample_rate, embed

"""
