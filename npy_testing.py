import numpy as np

def process_video(dataset):
    # print the shape of the dataset
    print(dataset.shape)

    # print the shape of the video
    print(dataset[0].shape)

    # print the shape of the label
    print(dataset[1].shape)
    
    print()
    
ds0 = np.load('dataset_p0.npy', allow_pickle=True)
# ds1 = np.load('dataset_p1.npy', allow_pickle=True)
# ds2 = np.load('dataset_p2.npy', allow_pickle=True)

process_video(ds0)
# process_video(ds1)
# process_video(ds2)