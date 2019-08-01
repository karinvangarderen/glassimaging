import fastr
import argparse

def create_network():
    network = fastr.create_network(id="applynetwork")

    source_t1 = network.create_source('NiftiImageFileCompressed', id='T1')
    source_t2 = network.create_source('NiftiImageFileCompressed', id='T2')
    source_t1Gd = network.create_source('NiftiImageFileCompressed', id='T1GD')
    source_flair = network.create_source('NiftiImageFileCompressed', id='FLAIR')
    source_model = network.create_source('Model', id='model')
    source_config = network.create_source('JSONFile', id='config')
    source_brainmask = network.create_source('NiftiImageFileCompressed', id='MASK')

    apply = network.create_node("glassimaging/SegmentTumor:1.0", tool_version='1.0', id='segment')

    source_t1.output >> apply.inputs['t1']
    source_t2.output >> apply.inputs['t2']
    source_t1Gd.output >> apply.inputs['t1gd']
    source_flair.output >> apply.inputs['flair']
    source_model.output >> apply.inputs['model']
    source_config.output >> apply.inputs['config']
    source_brainmask.output >> apply.inputs['brainmask']

    sink = network.create_sink('NiftiImageFileCompressed', id='segmentation')

    apply.outputs['seg'] >> sink.input




    return network

def main():
    network = create_network()

    network.draw()

    source_data = {'T1': "vfs://data/EGD-preprocessed/EGD-0016/Radiology_E47004/t1.nii.gz",
                  'T2': "vfs://data/EGD-preprocessed/EGD-0016/Radiology_E47004/t2.nii.gz",
                    'FLAIR':"vfs://data/EGD-preprocessed/EGD-0016/Radiology_E47004/flair.nii.gz",
                    'T1GD': "vfs://data/EGD-preprocessed/EGD-0016/Radiology_E47004/t1gd.nii.gz",
                    'MASK': "vfs://data/EGD-preprocessed/EGD-0016/Radiology_E47004/brainmask.nii.gz",
                   'config': 'vfs://home/glassimaging/config/apply_single.json',
                   'model': 'vfs://tmp/20190731123251_test/train_unet/model.pt',
                   }
    sink_data = {
        'segmentation': 'vfs://tmp/' + 'fastr-seg{ext}',
    }

    network.execute(source_data, sink_data)

if __name__ == '__main__':
    main()

