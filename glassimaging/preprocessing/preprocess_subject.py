
import json
import argparse
from glassimaging.preprocessing.coregister import create_network_egd

TYPES = ['FLAIR', 'T1', 'T1GD', 'T2']

def process_experiment(xnathost, subject, file_source, file_sink = None):

    xnat_replace_http = xnathost.replace('https://', 'xnat://')
    if file_sink == None:
        file_location = "vfs://home/EGD-preprocessed/{}/".format(subject) + "{sample_id}/"
    else:
        file_location = file_sink

    network = create_network_egd(segmentation=False)
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
        'corrected_flair': file_location + 'flair_corrected{ext}',
        #'transform_file_seg': file_location + 'transform_seg{ext}',
        #'transform_result_seg': file_location + 'seg{ext}',
    }
    source_data = {'T1': {},
                   'T2': {},
                   'T1GD': {},
                   'FLAIR': {},
                   'parameters': 'vfs://home/glassimaging/glassimaging/preprocessing/elastix_parameters.txt',
                   'parameters_seg': 'vfs://home/glassimaging/glassimaging/preprocessing/elastix_parameters_seg.txt',
                   'SEG': {},
                   'IMSEG': {}
                   }

    with open(file_source.format(subject), 'r+') as f:
        found_scans = json.load(f)

    for exp in found_scans:
            res = found_scans[exp]['scans']
            for type in TYPES:
                if type in res:
                    source_data[type][exp] = res[type]['uri'].replace('https://', 'xnat://') + '/resources/NIFTI/files/image.nii.gz'
                else:
                    source_data[type][exp] = ''
            if 'SEG' in res:
                source_data['SEG'][exp] = res['SEG']['uri'].replace('https://', 'xnat://') + '/resources/MASKS/files/tumor_MW.nii.gz'
                source_data['IMSEG'][exp] = res['SEG']['uri'].replace('https://', 'xnat://') + '/resources/MASKS/files/tumor_MW_image.nii.gz'
            else:
                source_data['SEG'][exp] = ''
                source_data['IMSEG'][exp] = ''


    network.execute(source_data, sink_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch available scans for a subject.')
    parser.add_argument('xnathost', help='url of xnat.')
    parser.add_argument('subject', help='subject name.')
    parser.add_argument('filename', help='json output filename.')

    args = parser.parse_args()
    process_experiment(args.xnathost, args.subject, args.filename)
