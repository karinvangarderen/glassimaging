import pandas as pd
import xnat
import json
import argparse

TYPES = ['FLAIR', 'T1', 'T1GD', 'T2']

def get_subjects(id_file):
    subj = pd.read_csv(id_file, header=None, names = ['id'])
    res = [s.replace("\t", "") for s in subj['id'].values]
    return res

def process_subject_xnat(xnat_subject, result_filename):
    data_to_download = {}
    for exp in xnat_subject.experiments:
        experiment = xnat_subject.experiments[exp]
        data_to_download[exp] = {
            'scans': {},
            'date': experiment.date.strftime("%d-%m-%Y")
        }
        scans_found = {}
        for sc in experiment.scans:
            scan = experiment.scans[sc]
            try:
                fields_file = scan.resources['FIELDS'].files[-1]
                with fields_file.open() as f:
                    data = json.load(f)
                    type = data['scan_type']
                    if type in TYPES:
                        if type not in scans_found:
                            scans_found[type] = {
                                'uri': scan.external_uri(),
                                'N_slices': data['N_slices']
                            }
                        elif data['N_slices'] > scans_found[type]['N_slices']:
                            scans_found[type] = {
                                'uri': scan.external_uri(),
                                'N_slices': data['N_slices']
                            }

            except KeyError as e:
                print('{} failed'.format(sc))
                print(xnat_subject.label)
                print(e)

            if 'MASKS' in scan.resources:
                for f in scan.resources['MASKS'].files:
                    if f == 'tumor_MW.nii.gz':
                        scans_found['SEG'] = {
                                                 'uri': scan.external_uri()
                                             }

        data_to_download[exp]['scans'] = scans_found
        print(scans_found)

        with open(result_filename, 'w+') as f:
            json.dump(data_to_download, f)
    return data_to_download

def process_subject_id(xnat_host, xnat_project, subject, result_filename):
    with xnat.connect(xnat_host) as connection:
        xnat_project = connection.projects[xnat_project]
        xnat_subject = xnat_project.subjects[subject]
        filename = result_filename.format(subject)
        process_subject_xnat(xnat_subject, filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch available scans for a subject.')
    parser.add_argument('xnathost', help='url of xnat.')
    parser.add_argument('subject', help='subject name.')
    parser.add_argument('project', help='xnat project.')
    parser.add_argument('filename', help='json output filename.')

    args = parser.parse_args()
    process_subject_id(args.xnathost, args.project, args.subject, args.filename)
