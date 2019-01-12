import numpy as np

# Segmentation for Training 
def make_segments(mels,labels, COL_SIZE = 120):
    '''
    Makes segments of mel and attaches them to the labels
    :param mels: list of mels
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    for mel,label in zip(mels,labels):
        for start in range(0, int(mel.shape[1] / COL_SIZE)):
            segments.append(mel[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return (segments, seg_labels)

# Segmentation for testing
def segment_one(mel, COL_SIZE = 120):
    '''
    Creates segments from on mel image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mel (numpy array): mel array
    :return (numpy array): Segmented mel array
    '''
    segments = []
    for start in range(0, int(mel.shape[1] / COL_SIZE)):
        segments.append(mel[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_mels(X_train, COL_SIZE = 120):
    '''
    Creates segmented mels from X_train
    :param X_train: list of mels
    :return: segmented mels
    '''
    segmented_mels = []
    for mel in X_train:
        segmented_mels.append(segment_one(mel, COL_SIZE))
    return(segmented_mels)
