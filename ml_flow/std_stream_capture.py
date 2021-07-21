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
import sys
import re
from io import StringIO


class StdStreamCapture(object):
    # https://stackoverflow.com/questions/19296667/remove-ansi-color-codes-from-a-text-file-using-bash
    remove_colour_codes_regexp = re.compile(r'\x1B\[(([0-9]{1,2})?(;)?([0-9]{1,2})?)?[mKHfJ]')

    def __init__(self):
        self.stream = StringIO()
        self.old_stdout_write = None
        self.old_stderr_write = None

    def __enter__(self):
        self.old_stderr_write = sys.stderr.write
        self.old_stdout_write = sys.stdout.write

        def patch_stderr_write(text):
            self.old_stderr_write(text)
            stripped = self.remove_colour_codes_regexp.sub("", text)
            if not stripped.startswith('\r'):
                # avoid needing to seek and re-write in StringIO, just skip write calls that
                # start with \r
                self.stream.write(stripped)

        def patch_stdout_write(text):
            self.old_stdout_write(text)
            stripped = self.remove_colour_codes_regexp.sub("", text)
            if not stripped.startswith('\r'):
                # avoid needing to seek and re-write in StringIO, just skip write calls that
                # start with \r
                self.stream.write(stripped)

        sys.stdout.write = patch_stdout_write
        sys.stderr.write = patch_stderr_write
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.write = self.old_stdout_write
        sys.stderr.write = self.old_stderr_write
        return self

    def read_all(self):
        loc = self.stream.tell()
        self.stream.seek(0)
        data = self.stream.read()
        self.stream.seek(loc)
        return data