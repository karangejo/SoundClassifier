import recAudio
import beatSlicer
import os


def record_label_from_jack(dataset_dir):
    print("You will have 15 seconds to record your training samples after you press enter!!")
    label = input("What is the label name? :")
    label_dir = dataset_dir+label
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
        print("Directory " , label ,  " Created ")
    else:
        print("Directory " , label ,  " already exists")

    print("RECORDING! You have 15 seconds to record your training samples")
    samplefile = label_dir+'/'+label+'.wav'
    recAudio.rec_audio_from_jack(samplefile,'system:capture_1',15)
    beatSlicer.slice_audio_sample_no_env(samplefile,label_dir)
    os.remove(samplefile)
    print("All finished!")

    
