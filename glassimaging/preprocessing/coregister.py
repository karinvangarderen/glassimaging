import fastr
import argparse

def create_network_egd():
    network = fastr.create_network(id="preprocess_glioma_egd")

    source_t1 = network.create_source('NiftiImageFileCompressed', id='T1')
    source_t2 = network.create_source('NiftiImageFileCompressed', id='T2')
    source_t1Gd = network.create_source('NiftiImageFileCompressed', id='T1GD')
    source_flair = network.create_source('NiftiImageFileCompressed', id='FLAIR')

    bet_node = network.create_node('fsl/Bet:5.0.9', tool_version='0.2', id='bet')
    bet_constant_eye = [True] >> bet_node.inputs['eye_cleanup']
    bet_constant_robust = [True] >> bet_node.inputs['robust_estimation']

    link_t1_bet = source_t1.output >> bet_node.inputs['image']
    link_t2_bet = source_t2.output >> bet_node.inputs['T2_image']

    node_resample = network.create_node('custom/resample:0.1', tool_version='0.1', id='resample')
    link_img_resample = source_t1.output >> node_resample.inputs['image']
    link_mask_resample = bet_node.outputs['mask_image'] >> node_resample.inputs['mask']
    sink_resample = network.create_sink('NiftiImageFileCompressed', id='resampled_t1')
    link_resample_sink = node_resample.outputs['image_resampled'] >> sink_resample.input

    sink_mask = network.create_sink('NiftiImageFileCompressed', id='bet_mask')
    link_mask_sink = node_resample.outputs['mask_resampled'] >> sink_mask.input

    source_elastix_params = network.create_source('ElastixParameterFile', id='parameters')

    create_coregister_transform(network, source_t1Gd.output, node_resample, source_elastix_params, 't1gd')
    create_coregister_transform(network, source_t2.output, node_resample, source_elastix_params, 't2')
    create_coregister_transform(network, source_flair.output, node_resample, source_elastix_params, 'flair')
    return network

def create_coregister_transform(network, source_image, source_baseline, source_parameters, name, transform_image=None):
    coregister = network.create_node('elastix/Elastix:4.8', tool_version='0.2', id='coregister_{}'.format(name))
    sink_transform = network.create_sink('ElastixTransformFile', id='transform_file_{}'.format(name))
    link_transform_sink = coregister.outputs['transform'] >> sink_transform.input
    link_parameters_registration = source_parameters.output >> coregister.inputs['parameters']
    link_convert_coregister = source_image >> coregister.inputs['moving_image']
    link_baseline_coregister = source_baseline.outputs['image_resampled'] >> coregister.inputs['fixed_image']
    transformix = network.create_node('elastix/Transformix:4.8', tool_version='0.2', id='transformix_{}'.format(name))
    link_coregister_transformix = coregister.outputs['transform'] >> transformix.inputs['transform']
    if transform_image is None:
        link_image_transformix = source_image >> transformix.inputs['image']
    else:
        transform_image >> transformix.inputs['image']
    sink_transformix = network.create_sink('NiftiImageFileCompressed', id='transform_result_{}'.format(name))
    link_cast_sink = transformix.outputs['image'] >> sink_transformix.input
    return



def main(resultpath):
    network = create_network_egd()

    network.draw()

    scans = "xnat://bigr-rad-xnat.erasmusmc.nl/data/archive/projects/EGD/subjects/Radiology_S17039/experiments/Radiology_E48799/scans/{}/resources/NIFTI/files/image.nii.gz"
    file_location = resultpath
    source_data = {'T1': scans.format('3'),
                  'T2': scans.format('8'),
                    'FLAIR': scans.format('6'),
                    'T1GD': scans.format('9'),
                   'parameters': 'vfs://home/glassimaging/glassimaging/preprocessing/elastix_parameters.txt',
                   'parameters_seg': 'vfs://home/glassimaging/glassimaging/preprocessing/elastix_parameters_seg.txt'}
    sink_data = {
        'resampled_t1': file_location + 't1{ext}',
        'bet_mask': file_location + 'brainmask{ext}',
        'transform_file_t1gd': file_location + 'transform_t1gd{ext}',
        'transform_file_t2': file_location + 'transform_t2{ext}',
        'transform_file_flair': file_location + 'transform_flair{ext}',
        'transform_result_t1gd': file_location + 't1gd{ext}',
        'transform_result_t2': file_location + 't2{ext}',
        'transform_result_flair': file_location + 'flair{ext}',
    }

    network.execute(source_data, sink_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test coregistration.')
    parser.add_argument('resultpath', help='directory to store results.')

    args = parser.parse_args()
    main(args.resultpath)

