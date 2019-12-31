import fastr
import argparse
import os
import itertools

def create_network_longitudinal():
    network = fastr.create_network(id="preprocess_glioma_egd")

    source_t1_fixed = network.create_source('NiftiImageFileCompressed', id='T1_fixed_in')
    source_t2_fixed = network.create_source('NiftiImageFileCompressed', id='T2_fixed_in')
    source_t1Gd_fixed = network.create_source('NiftiImageFileCompressed', id='T1GD_fixed_in')
    source_flair_fixed = network.create_source('NiftiImageFileCompressed', id='FLAIR_fixed_in')
    source_seg_fixed = network.create_source('NiftiImageFileCompressed', id='seg_fixed_in')
    source_brainmask_fixed = network.create_source('NiftiImageFileCompressed', id='brainmask_fixed_in')

    source_t1_moving = network.create_source('NiftiImageFileCompressed', id='T1_moving_in')
    source_t2_moving = network.create_source('NiftiImageFileCompressed', id='T2_moving_in')
    source_t1Gd_moving = network.create_source('NiftiImageFileCompressed', id='T1GD_moving_in')
    source_flair_moving = network.create_source('NiftiImageFileCompressed', id='FLAIR_moving_in')
    source_seg_moving = network.create_source('NiftiImageFileCompressed', id='seg_moving_in')
    source_brainmask_moving = network.create_source('NiftiImageFileCompressed', id='brainmask_moving_in')

    sink_t1_fixed = network.create_sink('NiftiImageFileCompressed', id='t1_fixed')
    source_t1_fixed.output >> sink_t1_fixed.input
    sink_t2_fixed = network.create_sink('NiftiImageFileCompressed', id='t2_fixed')
    source_t2_fixed.output >> sink_t2_fixed.input
    sink_t1Gd_fixed = network.create_sink('NiftiImageFileCompressed', id='t1gd_fixed')
    source_t1Gd_fixed.output >> sink_t1Gd_fixed.input
    sink_flair_fixed = network.create_sink('NiftiImageFileCompressed', id='flair_fixed')
    source_flair_fixed.output >> sink_flair_fixed.input
    sink_seg_fixed = network.create_sink('NiftiImageFileCompressed', id='seg_fixed')
    source_seg_fixed.output >> sink_seg_fixed.input
    sink_brainmask_fixed = network.create_sink('NiftiImageFileCompressed', id='brainmask_fixed')
    source_brainmask_fixed.output >> sink_brainmask_fixed.input

    source_elastix_params = network.create_source('ElastixParameterFile', id='parameters')

    limit = fastr.core.resourcelimit.ResourceLimit(memory='3G')
    coregister = network.create_node('elastix/Elastix:4.8', tool_version='0.2', id='coregister', resources=limit)
    sink_transform = network.create_sink('ElastixTransformFile', id='transform_file')
    coregister.outputs['transform'] >> sink_transform.input
    source_elastix_params.output >> coregister.inputs['parameters']
    source_t1Gd_moving.output >> coregister.inputs['moving_image']
    source_t1Gd_fixed.output>> coregister.inputs['fixed_image']

    transformix_t1 = create_transform(network, source_t1_moving, 't1', coregister.outputs['transform'])
    transformix_t1gd = create_transform(network, source_t1Gd_moving, 't1gd', coregister.outputs['transform'])
    transformix_t2 = create_transform(network, source_t2_moving, 't2', coregister.outputs['transform'])
    transformix_flair = create_transform(network, source_flair_moving, 'flair', coregister.outputs['transform'])

    transformix_seg = create_transform_seg(network, source_seg_moving, 'seg', coregister.outputs['transform'])
    transformix_brainmask = create_transform_seg(network, source_brainmask_moving, 'brainmask', coregister.outputs['transform'])

    return network

def create_transform(network, source_node, name, transform_output):
    transformix = network.create_node('elastix/Transformix:4.8', tool_version='0.2', id='transformix_{}'.format(name))
    transform_output >> transformix.inputs['transform']
    source_node.output >> transformix.inputs['image']
    sink_transformix = network.create_sink('NiftiImageFileCompressed', id='{}_moving'.format(name))
    transformix.outputs['image'] >> sink_transformix.input
    return transformix

def create_transform_seg(network, source_node, name, transform_output):
    transformix = network.create_node('elastix/Transformix:4.8', tool_version='0.2', id='transformix_{}'.format(name))
    node_edit_transform = network.create_node('elastixtools/EditElastixTransformFile:0.1', tool_version='0.1', id='edit_transform_{}'.format(name))
    transform_output >> node_edit_transform.inputs['transform']
    ["FinalBSplineInterpolationOrder=0"] >> node_edit_transform.inputs['set']
    node_edit_transform.outputs['transform'] >> transformix.inputs['transform']
    source_node.output >> transformix.inputs['image']
    sink_transformix = network.create_sink('NiftiImageFileCompressed', id='{}_moving'.format(name))
    transformix.outputs['image'] >> sink_transformix.input
    return transformix


def main(resultpath, inputpath, subject):
    network = create_network_longitudinal()

    network.draw()

    source_data = {'T1_fixed_in': {},
                   'T2_fixed_in': {},
                   'FLAIR_fixed_in': {},
                   'T1GD_fixed_in': {},
                   'T1_moving_in': {},
                   'T2_moving_in': {},
                   'FLAIR_moving_in': {},
                   'T1GD_moving_in': {},
                   'seg_fixed_in': {},
                   'seg_moving_in': {},
                   'brainmask_fixed_in': {},
                   'brainmask_moving_in': {},
                   'parameters': 'vfs://home/glassimaging/glassimaging/preprocessing/elastix_parameters_nonlinear.txt'
                   }

    directories = []
    for d in os.listdir(inputpath):
        if subject in d:
            directories.append(d)

    for exp_combination in itertools.combinations(directories, 2):
        date_0 = exp_combination[0].split(subject)[1]
        date_1 = exp_combination[1].split(subject)[1]
        sample_name = subject + '_' + date_0 + '_' +  date_1

        source_data['T1_fixed_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[0], 't1.nii.gz'))
        source_data['T2_fixed_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[0], 't2.nii.gz'))
        source_data['FLAIR_fixed_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[0], 'flair.nii.gz'))
        source_data['T1GD_fixed_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[0], 't1gd.nii.gz'))
        source_data['seg_fixed_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[0], 'original_seg.nii.gz'))
        source_data['brainmask_fixed_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[0], 'brainmask.nii.gz'))

        source_data['T1_moving_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[1], 't1.nii.gz'))
        source_data['T2_moving_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[1], 't2.nii.gz'))
        source_data['FLAIR_moving_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[1], 'flair.nii.gz'))
        source_data['T1GD_moving_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[1], 't1gd.nii.gz'))
        source_data['seg_moving_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[1], 'original_seg.nii.gz'))
        source_data['brainmask_moving_in'][sample_name] = fastr.vfs.path_to_url(
            os.path.join(inputpath, exp_combination[1], 'brainmask.nii.gz'))


    sink_data = {
        't1_fixed': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/t1_fixed{ext}')),
        't2_fixed': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/t2_fixed{ext}')),
        't1gd_fixed': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/t1gd_fixed{ext}')),
        'flair_fixed': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/flair_fixed{ext}')),
        'seg_fixed': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/seg_fixed{ext}')),
        'brainmask_fixed': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/brainmask_fixed{ext}')),
        't1_moving': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/t1_moving{ext}')),
        't2_moving': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/t2_moving{ext}')),
        'flair_moving': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/flair_moving{ext}')),
        't1gd_moving': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/t1gd_moving{ext}')),
        'seg_moving': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/seg_moving{ext}')),
        'brainmask_moving': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/brainmask_moving{ext}')),
        'transform_file': fastr.vfs.path_to_url(os.path.join(resultpath, '{sample_id}/transform{ext}'))
    }

    network.execute(source_data, sink_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test coregistration.')
    parser.add_argument('--outdir', help='directory to store results.')
    parser.add_argument('--indir', help='directory containing input files.')
    parser.add_argument('--subject', help='EGD id for subject')

    args = parser.parse_args()
    main(args.outdir, args.indir, args.subject)

