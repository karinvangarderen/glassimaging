# Copyright 2011-2014 Biomedical Imaging Group Rotterdam, Departments of
# Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import fastr
from fastr.data import url
from fastr.datatypes import URLType
#from fastr.utils.checksum import checksum_directory


class DicomImageFile(URLType):
    description = 'Dicom Image directory'
    extension = ''
    magic_data = 'DICM'

    def is_dicom(self, filepath):
        if not os.path.isfile(filepath):
            return False

        with open(filepath, 'r') as fh_in:
            fh_in.seek(128)
            preamble = fh_in.read(4)
            return preamble == self.magic_data

    #def checksum(self):
    #    value = self.value
        #if url.isurl(self.value):
        #    value = url.get_path_from_url(value)

        #return checksum_directory(value)

    def _validate(self):
        value = self.value

        if url.isurl(self.value):
            value = url.get_path_from_url(value)

        try:
            if not os.path.isdir(value):
                return False

            contents = os.listdir(value)
            return any(self.is_dicom(os.path.join(value, x)) for x in contents)
        except ValueError:
            return False

    def action(self, name):
        if name is None:
            pass
        elif name == "ensure":
            if url.isurl(self.value):
                dir_name = url.get_path_from_url(self.value)
            else:
                dir_name = self.value

            fastr.log.debug('ensuring {} exists.'.format(dir_name))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        else:
            fastr.log.warning("unknown action")
