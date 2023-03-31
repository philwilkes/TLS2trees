import sys
import os
import argparse
import pickle

# from fsct.run_tools import FSCT
from fsct.other_parameters import other_parameters
from fsct.tools import dict2class
from fsct.preprocessing import Preprocessing
from fsct.inference import SemanticSegmentation
from fsct.segmentation import Segmentation

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--point-cloud', '-p', default='', type=str, help='path to point cloud')
    parser.add_argument('--params', type=str, default='', help='path to pickled parameter file')
    parser.add_argument('--odir', type=str, default='.', help='output directory')

    # run or redo steps
    parser.add_argument('--step', type=int, default=3, help='which process to run to')
    parser.add_argument('--redo', type=int, default=None, help='which process to run to')

    # if applying to tiled data
    parser.add_argument('--tile-index', default='', type=str, help='path to tile index in space delimited format "TILE X Y"')
    parser.add_argument('--buffer', default=0, type=float, help='included data from neighbouring tiles')
    
    # Set these appropriately for your hardware.
    parser.add_argument('--batch_size', default=10, type=int, help="If you get CUDA errors, try lowering this.")
    parser.add_argument('--num_procs', default=10, type=int, help="Number of CPU cores you want to use. If you run out of RAM, lower this.")

    parser.add_argument('--keep-npy', action='store_true', help="Keeps .npy files used for segmentation after inference is finished.")

    parser.add_argument('--is-wood', default=1, type=float, help='a probability above which points are classified as wood')
    parser.add_argument('--model', default=None, type=str, help='path to candidate model')
    
                       
    parser.add_argument('--output_fmt', default='ply', help="file type of output")
    parser.add_argument('--verbose', action='store_true', help="print stuff")

    params = parser.parse_args()
    
    ### sanity checks ###
    if params.point_cloud == '' and params.params == '':
        raise Exception('no input specified, use either --point-cloud or --params')
    
    if not os.path.isfile(params.point_cloud):
        if not os.path.isfile(params.params):
            raise Exception(f'no point cloud at {params.point_cloud}')
    
    if params.buffer > 0:
        if params.tile_index == '':
            raise Exception(f'buffer > 0 but no tile index specified, use --tile-index')
        if not os.path.isfile(params.tile_index):
            raise Exception(f'buffer > 0 but no tile index at {params.tile_index}')
    
    ### end sanity checks ###
   
    if os.path.isfile(params.params):
        p_space = pickle.load(open(params.params, 'rb'))
        for k, v in p_space.__dict__.items():
            # over ride saved parameters
            if k == 'params': continue
            if k == 'step': continue
            if k == 'redo': continue
            if k == 'batch_size': continue
            setattr(params, k, v)
    else:
        for k, v in other_parameters.items():
            if k == 'model' and params.model != None:
                setattr(params, k, params.model)
            elif k == 'is_wood' and params.is_wood < 1:
                setattr(params, k, params.is_wood)
            else:
                setattr(params, k, v)
        # i.e. if running for the first time - 
        # hack to add to params where used for 
        # recording which steps have been completed
        params.steps_completed = {0:False, 1:False}

    if params.redo != None:
        for k in params.steps_completed.keys():
            if k >= params.redo:
                params.steps_completed[k] = False

    if params.verbose:
        print('\n---- parameters used ----')
        for k, v in params.__dict__.items():
            if k == 'pc': v = '{} points'.format(len(v))
            if k == 'global_shift': v = v.values
            print('{:<35}{}'.format(k, v)) 

    if params.step >= 0 and not params.steps_completed[0]:
        params = Preprocessing(params)
        params.steps_completed[0] = True
        pickle.dump(params, open(os.path.join(params.odir, f'{params.basename}.params.pickle'), 'wb'))

    if params.step >= 1 and not params.steps_completed[1]:

        params = SemanticSegmentation(params)
        params.steps_completed[1] = True
        pickle.dump(params, open(os.path.join(params.odir, f'{params.basename}.params.pickle'), 'wb'))
