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

from fastr.core.version import Version
from fastr.datatypes import URLType


class ElastixLogFile(URLType):
    id = 'ElastixLogFile'
    name = 'ElastixLogFile'
    version = Version('1.0')
    description = 'Log file from Elastix'
    extension = 'log'

    def __eq__(self, other):
        """
        Log files are equal by default as long as they are both valid

        :param ElastixLogFile other: other to compare against
        :return: equality flag
        """
        return self.valid and other.valid
