import time
import rtmidi

# function that takes a list of classes and midi notes
# and returns a dictionary to convert between the two
def classes_to_midi_dict(classes,midi_notes):
    classes_to_midi = {}
    for index, class in enumerate(classes):
        classes_to_midi.update({class,midi_notes[index]})
    return(classes_to_midi)

# a function that plays a midi sequnce given an array of time intervals for beat slices (in seconds)
# and their classifications as well as a dictionary that can get the desired notes from the classifications
def midi_sequence_from_classified_audio(slice_output,classifications,classes_to_midi):
    # start midi client
    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()

    # connect to first open port
    if available_ports:
        midiout.open_port(0)
    else:
        midiout.open_virtual_port("My virtual output")

    # send out our sequence
    with midiout:
        for i in len(classifications):
            delay = 0.01
            note = classes_to_midi[classifications[i]]
            print(note)
            note_on = [0x90, note, 127] # channel 1, note, velocity 112
            note_off = [0x80, note, 0]
            midiout.send_message(note_on)
            time.sleep(delay)
            midiout.send_message(note_off)
            time.sleep(slice_output[i] - delay)
            print(slice_output[i])
