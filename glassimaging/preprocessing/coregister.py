import fastr
import argparse

def create_network_egd(apply_model=False, segmentation=False):
    network = fastr.create_network(id="preprocess_glioma_egd")

    source_t1 = network.create_source('NiftiImageFileCompressed', id='T1')
    source_t2 = network.create_source('NiftiImageFileCompressed', id='T2')
    source_t1Gd = network.create_source('NiftiImageFileCompressed', id='T1GD')
    source_flair = network.create_source('NiftiImageFileCompressed', id='FLAIR')

    limit = fastr.core.resourcelimit.ResourceLimit(memory='3G')
    bet_node = network.create_node('fsl/Bet:5.0.9', tool_version='0.2', id='bet', resources=limit)
    bet_constant_eye = [True] >> bet_node.inputs['eye_cleanup']
    bet_constant_robust = [True] >> bet_node.inputs['robust_estimation']

    link_t1_bet = source_t1.output >> bet_node.inputs['image']
    link_t2_bet = source_t2.output >> bet_node.inputs['T2_image']


    node_resample = network.create_node('custom/resample:0.1', tool_version='0.1', id='resample', resources=limit)
    link_img_resample = source_t1.output >> node_resample.inputs['image']
    link_mask_resample = bet_node.outputs['mask_image'] >> node_resample.inputs['mask']
    sink_resample = network.create_sink('NiftiImageFileCompressed', id='resampled_t1')
    link_resample_sink = node_resample.outputs['image_resampled'] >> sink_resample.input

    sink_mask = network.create_sink('NiftiImageFileCompressed', id='bet_mask')
    link_mask_sink = node_resample.outputs['mask_resampled'] >> sink_mask.input

    source_elastix_params = network.create_source('ElastixParameterFile', id='parameters')

    transform_t1gd = create_coregister_transform(network, source_t1Gd.output, node_resample, source_elastix_params, 't1gd')
    transform_t2 = create_coregister_transform(network, source_t2.output, node_resample, source_elastix_params, 't2')
    transform_flair = create_coregister_transform(network, source_flair.output, node_resample, source_elastix_params, 'flair')

    limit_bias = fastr.core.resourcelimit.ResourceLimit(memory='5G')
    node_biasfield = network.create_node('n4/N4:1.6', tool_version='0.1', id='biasfield', resources=limit_bias)
    transform_flair.outputs['image'] >> node_biasfield.inputs['image']
    node_resample.outputs['mask_resampled'] >> node_biasfield.inputs['mask']

    sink_bias_corrected_flair = network.create_sink('NiftiImageFileCompressed', id='corrected_flair')
    link_cast_sink = node_biasfield.outputs['image'] >> sink_bias_corrected_flair.input

    if apply_model:
        source_model = network.create_source('Model', id='MODEL')
        source_config = network.create_source('JsonFile', id='MODEL_CONFIG')

        apply_model = network.create_node("glassimaging/SegmentTumor:1.0", tool_version='1.0', id='segment')

        node_resample.outputs['image_resampled'] >> apply_model.inputs['t1']
        transform_t2.outputs['image'] >> apply_model.inputs['t2']
        transform_t1gd.outputs['image'] >> apply_model.inputs['t1gd']
        transform_flair.outputs['image'] >> apply_model.inputs['flair']
        source_model.output >> apply_model.inputs['model']
        source_config.output >> apply_model.inputs['config']
        node_resample.outputs['mask_resampled'] >> apply_model.inputs['brainmask']

        sink = network.create_sink('NiftiImageFileCompressed', id='segmentation')

        apply_model.outputs['seg'] >> sink.input

    if segmentation:
        source_elastix_params_seg = network.create_source('ElastixParameterFile', id='parameters_seg')
        source_seg = network.create_source('NiftiImageFileCompressed', id='SEG')
        source_imseg = network.create_source('NiftiImageFileCompressed', id='IMSEG')
        create_coregister_transform_seg(network, source_imseg.output, node_resample, source_seg.output,
                                        source_elastix_params_seg, 'seg')

    return network

def create_coregister_transform(network, source_image, source_baseline, source_parameters, name, transform_image=None):
    limit = fastr.core.resourcelimit.ResourceLimit(memory='3G')
    coregister = network.create_node('elastix/Elastix:4.8', tool_version='0.2', id='coregister_{}'.format(name), resources=limit)
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
    return transformix

def create_coregister_transform_seg(network, source_image, source_baseline, source_seg, source_parameters, name):
    coregister = network.create_node('elastix/Elastix:4.8', tool_version='0.2', id='coregister_{}'.format(name))
    sink_transform = network.create_sink('ElastixTransformFile', id='transform_file_{}'.format(name))
    link_transform_sink = coregister.outputs['transform'] >> sink_transform.input
    link_parameters_registration = source_parameters.output >> coregister.inputs['parameters']
    link_convert_coregister = source_image >> coregister.inputs['moving_image']
    link_baseline_coregister = source_baseline.outputs['image_resampled'] >> coregister.inputs['fixed_image']
    transformix = network.create_node('elastix/Transformix:4.8', tool_version='0.2', id='transformix_{}'.format(name))
    link_coregister_transformix = coregister.outputs['transform'] >> transformix.inputs['transform']
    link_image_transformix = source_seg >> transformix.inputs['image']
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
                   'parameters_seg': 'vfs://home/glassimaging/glassimaging/preprocessing/elastix_parameters_seg.txt',
                   'MODEL': ' vfs://home/applymodel/model.pt',
                   'MODEL_CONFIG': ' vfs://home/applymodel/config.json'}
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
        'segmentation': file_location + 'seg{ext}',
    }

    network.execute(source_data, sink_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test coregistration.')
    parser.add_argument('resultpath', help='directory to store results.')

    args = parser.parse_args()
    main(args.resultpath)

