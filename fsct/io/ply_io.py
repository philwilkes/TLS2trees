import pandas as pd
import numpy as np
import sys

def read_ply(fp, newline=None):

    if (sys.version_info > (3, 0)):
        open_file = open(fp, encoding='ISO-8859-1', newline='\n' if sys.platform == 'win32' else None)
    else:
        open_file = open(fp)

    with open_file as ply:
 
        length = 0
        prop = []
        dtype_map = {'uint16':'uint16', 'uint8':'uint8', 'double':'d', 'float64':'f8', 
                     'float32':'f4', 'float': 'f4', 'uchar': 'B', 'int':'i'}
        dtype = []
        fmt = 'binary'
    
        for i, line in enumerate(ply.readlines()):
            length += len(line)
            if i == 1:
                if 'ascii' in line:
                    fmt = 'ascii' 
            if 'element vertex' in line: N = int(line.split()[2])
            if 'property' in line: 
                dtype.append(dtype_map[line.split()[1]])
                prop.append(line.split()[2])
            if 'element face' in line:
                raise Exception('.ply appears to be a mesh')
            if 'end_header' in line: break
    
        ply.seek(length)

        if fmt == 'binary':
            arr = np.fromfile(ply, dtype=','.join(dtype))
        else:
            arr = np.loadtxt(ply)
        df = pd.DataFrame(data=arr)
        df.columns = prop

    return df

def write_ply(output_name, pc, comments=[]):

    cols = ['x', 'y', 'z']
    pc[['x', 'y', 'z']] = pc[['x', 'y', 'z']].astype('f8')

    with open(output_name, 'w') as ply:

        ply.write("ply\n")
        ply.write('format binary_little_endian 1.0\n')
        ply.write("comment Author: Phil Wilkes\n")
        for comment in comments:
            ply.write("comment {}\n".format(comment))
        ply.write("obj_info generated with pcd2ply.py\n")
        ply.write("element vertex {}\n".format(len(pc)))
        ply.write("property float64 x\n")
        ply.write("property float64 y\n")
        ply.write("property float64 z\n")
        if 'red' in pc.columns:
            cols += ['red', 'green', 'blue']
            pc[['red', 'green', 'blue']] = pc[['red', 'green', 'blue']].astype('i')
            ply.write("property int red\n")
            ply.write("property int green\n")
            ply.write("property int blue\n")
        for col in pc.columns:
            if col in cols: continue
            try:
                pc[col] = pc[col].astype('f8')
                ply.write("property float64 {}\n".format(col))
                cols += [col]
            except:
                pass
        ply.write("end_header\n")

    with open(output_name, 'ab') as ply:
        ply.write(pc[cols].to_records(index=False).tobytes()) 
