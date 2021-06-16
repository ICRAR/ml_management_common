#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2021
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#
import logging
from io import StringIO


class StreamLoggerContext(object):

    def __init__(self):
        self.stream = StringIO()
        self.logger = logging.StreamHandler(self.stream)
        self.logger.setLevel(logging.INFO)
        self.logger.setFormatter(logging.Formatter('%(asctime)-15s:' + logging.BASIC_FORMAT))

    def __enter__(self):
        logging.root.addHandler(self.logger)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.root.removeHandler(self.logger)
        return self

    def write(self, target):
        target.write(bytes(self.read_all(), "utf-8"))

    def read_all(self):
        loc = self.stream.tell()
        self.stream.seek(0)
        data = self.stream.read()
        self.stream.seek(loc)
        return data