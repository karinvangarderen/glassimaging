
import json
import argparse
from glassimaging.preprocessing.coregister import create_network_egd

TYPES = ['FLAIR', 'T1', 'T1GD', 'T2']

def process_experiment(xnathost, subject, path_scans):

    xnat_replace_http = xnathost.replace('https://', 'xnat://')
    file_location = "vfs://home/EGD-preprocessed/{}/".format(subject) + "{sample_id}/"

    network = create_network_egd()
    network.draw()
    sink_data = {
        'resampled_t1': file_location + 't1{ext}',
        'bet_mask': file_location + 'brainmask{ext}',
        'transform_file_t1gd': file_location + 'transform_t1gd{ext}',
        'transform_file_t2': file_location + 'transform_t2{ext}',
        'transform_file_flair': file_location + 'transform_flair{ext}',
        'transform_result_t1gd': file_location + 't1gd{ext}',
        'transform_result_t2': file_location + 't2{ext}',
        'transform_result_flair': file_location + 'flair{ext}',
        'segmentation': file_location + 'seg{ext}',

    }
    source_data = {'T1': {},
                   'T2': {},
                   'T1GD': {},
                   'FLAIR': {},
                   'parameters': 'vfs://home/glassimaging/glassimaging/preprocessing/elastix_parameters.txt',
                   'parameters_seg': 'vfs://home/glassimaging/glassimaging/preprocessing/elastix_parameters_seg.txt',
                   'MODEL': 'vfs://home/applymodel/model.pt',
                   'MODEL_CONFIG': 'vfs://home/applymodel/config.json'}

    with open(path_scans.format(subject), 'r+') as f:
        found_scans = json.load(f)

    for exp in found_scans:
            res = found_scans[exp]
            for type in TYPES:
                if type in res:
                    source_data[type][exp] = xnat_replace_http + res[type]['uri'] + '/resources/NIFTI/files/image.nii.gz'
                else:
                    source_data[type][exp] = ''

    network.execute(source_data, sink_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test coregistration.')
    parser.add_argument('xnathost', help='url of xnat.')
    parser.add_argument('subject', help='subject name.')
    parser.add_argument('path', help='path of file containing found scans.')

    args = parser.parse_args()
    process_experiment(args.xnathost, args.subject, args.path)
