import fastr
import argparse

##Test file which skips the time-consuming BET step

def create_network_egd(apply_model=True, segmentation=True):
    network = fastr.create_network(id="ease_test")

    source_t1 = network.create_source('NiftiImageFileCompressed', id='T1')
    source_t2 = network.create_source('NiftiImageFileCompressed', id='T2')
    source_t1Gd = network.create_source('NiftiImageFileCompressed', id='T1GD')
    source_flair = network.create_source('NiftiImageFileCompressed', id='FLAIR')
    source_brainmask = network.create_source('NiftiImageFileCompressed', id='BRAINMASK')

    limit = fastr.core.resourcelimit.ResourceLimit(memory='3G')

    node_resample = network.create_node('glassimaging/resample:0.1', tool_version='0.1', id='resample', resources=limit)
    link_img_resample = source_t1.output >> node_resample.inputs['image']
    link_mask_resample = source_brainmask.output >> node_resample.inputs['mask']
    sink_resample = network.create_sink('NiftiImageFileCompressed', id='resampled_t1')
    link_resample_sink = node_resample.outputs['image_resampled'] >> sink_resample.input

    sink_mask = network.create_sink('NiftiImageFileCompressed', id='bet_mask')
    link_mask_sink = node_resample.outputs['mask_resampled'] >> sink_mask.input

    source_elastix_params = network.create_source('ElastixParameterFile', id='parameters')

    transform_t1gd = create_coregister_transform(network, source_t1Gd.output, node_resample, source_elastix_params, 't1gd')
    transform_t2 = create_coregister_transform(network, source_t2.output, node_resample, source_elastix_params, 't2')
    transform_flair = create_coregister_transform(network, source_flair.output, node_resample, source_elastix_params, 'flair')

    limit_bias = fastr.core.resourcelimit.ResourceLimit(memory='5G')
    out_flair = correct_biasfield(network, transform_flair.outputs['image'], node_resample.outputs['mask_resampled'], 'FLAIR', limit_bias)
    out_t1gd = correct_biasfield(network, transform_t1gd.outputs['image'], node_resample.outputs['mask_resampled'], 'T1GD', limit_bias)
    out_t2 = correct_biasfield(network, transform_t2.outputs['image'], node_resample.outputs['mask_resampled'], 'T2', limit_bias)
    out_t1 = correct_biasfield(network, node_resample.outputs['image_resampled'], node_resample.outputs['mask_resampled'], 'T1', limit_bias)


    if apply_model:
        source_model = network.create_source('Model', id='MODEL')
        source_config = network.create_source('JsonFile', id='MODEL_CONFIG')

        apply_model = network.create_node("glassimaging/SegmentTumor:1.0", tool_version='1.0', id='segment')

        out_t1 >> apply_model.inputs['t1']
        out_t2 >> apply_model.inputs['t2']
        out_t1gd >> apply_model.inputs['t1gd']
        out_flair >> apply_model.inputs['flair']
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

def preprocess_image(network, source_dicom, id):
    dcm2nii = network.create_node('dcm2niix/DicomToNifti:0.1', tool_version='0.1', id='dcm2nii_{}'.format(id))
    source_dicom.output >> dcm2nii.inputs['dicom_image']
    fsl2std = network.create_node('fsl/FSLReorient2Std:5.0.9', tool_version='0.1', id='fsl2std_{}'.format(id))
    dcm2nii.outputs['image'] >> fsl2std.inputs['image']
    return fsl2std.outputs['reoriented_image']

def correct_biasfield(network, image_output, mask_output, id, resourcelimit):
    node_biasfield = network.create_node('n4/N4:1.6', tool_version='0.1', id='biasfield_{}'.format(id), resources=resourcelimit)
    image_output >> node_biasfield.inputs['image']
    mask_output >> node_biasfield.inputs['mask']

    sink_bias_corrected_flair = network.create_sink('NiftiImageFileCompressed', id='out_{}'.format(id))
    link_cast_sink = node_biasfield.outputs['image'] >> sink_bias_corrected_flair.input
    return node_biasfield.outputs['image']



def main(resultpath):
    network = create_network_egd()

    network.draw()

    file_location = resultpath
    source_data = {
        'T1': {},
        'T2': {},
        'FLAIR': {},
        'T1GD': {},
        'BRAINMASK': {},
        'SEG': {},
        'IMSEG': {},
        'parameters': 'vfs://home/glassimaging/glassimaging/preprocessing/elastix_parameters.txt',
        'parameters_seg': 'vfs://home/glassimaging/glassimaging/preprocessing/elastix_parameters_seg.txt',
        'MODEL': 'vfs://home/applymodel/model.pt',
        'MODEL_CONFIG': 'vfs://home/applymodel/config.json'}
    for date in ['2000-01-01']:
        source_data['T1'][date] = 'vfs://home/EASE_test/EGD-test/{}/t1.nii.gz'.format(date),
        source_data['T1GD'][date] = 'vfs://home/EASE_test/EGD-test/{}/t1gd.nii.gz'.format(date),
        source_data['T2'][date] = 'vfs://home/EASE_test/EGD-test/{}/t2.nii.gz'.format(date),
        source_data['FLAIR'][date] = 'vfs://home/EASE_test/EGD-test/{}/flair.nii.gz'.format(date),
        source_data['BRAINMASK'][date] = 'vfs://home/EASE_test/EGD-test/{}/brainmask.nii.gz'.format(date),
        source_data['SEG'][date] = 'vfs://home/EASE_test/EGD-test/{}/seg.nii.gz'.format(date),
        source_data['IMSEG'][date] = 'vfs://home/EASE_test/EGD-test/{}/flair.nii.gz'.format(date),
    sink_data = {
        'resampled_t1': file_location + '/{sample_id}/' + 't1_resampled{ext}',
        'bet_mask': file_location + '/{sample_id}/' + 'brainmask{ext}',
        'transform_file_t1gd': file_location +'/{sample_id}/' + 'transform_t1gd{ext}',
        'transform_file_t2': file_location +'/{sample_id}/' + 'transform_t2{ext}',
        'transform_file_flair': file_location +'/{sample_id}/' + 'transform_flair{ext}',
        'out_T1GD': file_location +'/{sample_id}/' + 't1gd{ext}',
        'out_T2': file_location +'/{sample_id}/' + 't2{ext}',
        'out_FLAIR': file_location +'/{sample_id}/' + 'flair{ext}',
        'out_T1': file_location +'/{sample_id}/' + 't1{ext}',
        'segmentation': file_location +'/{sample_id}/' + 'seg{ext}',
        'transform_file_seg': file_location +'/{sample_id}/' + 'transform_seg{ext}',
        'transform_result_seg': file_location +'/{sample_id}/' + 'original_seg{ext}'
    }

    network.execute(source_data, sink_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test coregistration.')
    parser.add_argument('resultpath', help='directory to store results.')

    args = parser.parse_args()
    main(args.resultpath)

