import pandas as pd
import xnat
import json
import argparse
import os

TYPES = ['FLAIR', 'T1', 'T1GD', 'T2']
XNATHOST = "https://bigr-rad-xnat.erasmusmc.nl"
XNATPROJECT = 'EGD'

def get_list_of_scans(id_file, targetdir):
    df = pd.read_csv(id_file)
    data_to_download = {}
    log_download_filename = os.path.join(targetdir, 'log_downloaded.json')
    with xnat.connect(XNATHOST) as connection:
        xnat_project = connection.projects[XNATPROJECT]
        for index, row in df.iterrows():
            subject = row['Subject']
            xnat_subject = xnat_project.subjects[subject]
            experiment = xnat_subject.experiments[row['Exp']]
            if not subject in data_to_download:
                data_to_download[subject] = {}
            xnat_datestring = experiment.date.strftime("%d-%m-%Y")
            real_date = row['Real Date']
            data_to_download[row['Subject']][row['Real Date']] = {
                'scans': {},
                'xnat date': xnat_datestring
            }
            scans_found = {}
            for sc in experiment.scans:
                scan = experiment.scans[sc]
                try:
                    fields_file = scan.resources['FIELDS'].files[-1]
                    with fields_file.open() as f:
                        data = json.load(f)
                        type = data['scan_type']
                        print(type)
                        if type in TYPES:
                            if type not in scans_found:
                                scans_found[type] = {
                                    'uri': scan.external_uri(),
                                    'id': sc
                                }
                            elif data['Direction'] is 'Axial':
                                scans_found[type] = {
                                    'uri': scan.external_uri(),
                                    'id': sc
                                }

                except KeyError as e:
                    print('{} failed'.format(sc))
                    print(xnat_subject.label)
                    print(e)

                if 'MASKS' in scan.resources:
                    for f in scan.resources['MASKS'].files:
                        if f == 'tumor_MW.nii.gz':
                            scans_found['SEG'] = {
                                'uri': scan.external_uri(),
                                'id': sc
                            }

            data_to_download[subject][real_date]['scans'] = scans_found
            with open(log_download_filename, 'w+') as f:
                json.dump(data_to_download, f)

            if all(t in scans_found for t in TYPES + ['SEG']):
                downdir = os.path.join(targetdir, subject, real_date)
                os.makedirs(downdir)
                for t in TYPES:
                    experiment.scans[scans_found[t]['id']].resources['DICOM'].download_dir(os.path.join(downdir, t))
                experiment.scans[scans_found['SEG']['id']].resources['MASKS'].files['tumor_MW.nii.gz'].download(
                    os.path.join(downdir, 'seg.nii.gz'))
                experiment.scans[scans_found['SEG']['id']].resources['MASKS'].files['tumor_MW_image.nii.gz'].download(
                    os.path.join(downdir, 'seg_img.nii.gz'))
            

    return data_to_download

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch available scans for a subject.')
    parser.add_argument('idfile', help='file containing matched scans')
    parser.add_argument('targetdir', help='directory to download to')
    args = parser.parse_args()
    scans = get_list_of_scans(args.idfile, args.targetdir)
