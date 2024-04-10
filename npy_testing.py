import numpy as np

def process_video(dataset):
    # print the shape of the dataset
    print(f'dataset shape: {dataset.shape}')
    
    # print shape of the first entry
    print(f'shape of first entry: {dataset[0].shape}')
    
    # print shape of the video in the first entry
    print(f'shape of video in first entry: {dataset[0][0].shape}')
    
    # print shape of the label in the first entry
    print(f'shape of label in first entry: {dataset[0][1].shape}')
    
    # print the label in the first entry
    print(f'label in first entry: {dataset[0][1]}')
    
    print()
    
ds0 = np.load('dataset_p0.npy', allow_pickle=True)
# ds1 = np.load('dataset_p1.npy', allow_pickle=True)
# ds2 = np.load('dataset_p2.npy', allow_pickle=True)

process_video(ds0)
# process_video(ds1)
# process_video(ds2)