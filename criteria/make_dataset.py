import os
import argparse

def make_dataset(root: str):
    """
    Reads a directory with data.
    Returns a dataset as a list of tuples of paired image paths: (rgb_path, gt_path)
    """
    dataset = []

    # Our dir names
    gr_dir = 'generated'
    gt_dir = 'ground_truth'   
    
    # Get all the filenames from GroundTruth and Generated folder
    gt_imgList = os.listdir(os.path.join(root, gt_dir))
    gt_frames = sorted(gt_imgList, key= lambda x: int(x.split('.')[0]))
    # print('GroundTruth Nums:', len(gt_frames))
    gr_imgList = os.listdir(os.path.join(root, gr_dir))
    gr_frames = sorted(gr_imgList, key= lambda x: int(x.split('.')[0]))
    # print('Generated Nums:', len(gr_frames))
    
    N = 8
    for i in range(len(gt_frames)):
        # [0, 39]
        gt_path = os.path.join(root, gt_dir, gt_frames[i])
        
        for j in range(8):
            # [0, 319]
            gr_path = os.path.join(root, gr_dir, gr_frames[i * N + j])
            item = (gr_path, gt_path)
            # append to the list dataset
            dataset.append(item)

    return dataset

"""
def main(args):
    dataset = make_dataset(args.root)
    print('总的数据集对数：', len(dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='val_data', 
                        help='Data location')
                        
    args = parser.parse_args()
    main(args)
"""