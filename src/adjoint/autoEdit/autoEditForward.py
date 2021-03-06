#! /usr/bin/env python
"""
autoEdit - A Python tool to automatically edit a set of files
           according to the specified user rules:
G. Kenway
"""

# Import modules
import os, sys
import string
import re
# Specify file extension
EXT = '_d.f90'

DIR_ORI = sys.argv[1]
DIR_MOD = sys.argv[2]

# Specifiy the list of LINE ID's to find, what to replace and with what
patt_modules = re.compile(r'(\s*use\s*\w*)(_d)\s*')
patt_module = re.compile(r'\s*module\s\w*')
patt_subroutine = re.compile(r'\s*subroutine\s\w*')
patt_comment = re.compile(r'\s*!.*')
print "Directory of input source files  :", DIR_ORI
print "Directory of output source files :", DIR_MOD

useful_modules = ['solve_d']

for f in os.listdir(DIR_ORI):
    if f.endswith(EXT):
        # open original file in read mode
        file_object_ori = open(os.path.join(DIR_ORI,f),'r')
        print "\nParsing input file", file_object_ori.name

        # read to whole file to string and reposition the pointer
        # at the first byte for future reading
        all_src = file_object_ori.read()
        file_object_ori.seek(0)

        # First we want to determine if it is a 'useful' module or a
        # 'useless' module. A useful module is one that has
        # subroutines in it.
        isModule = False
        hasSubroutine = False
        for line in file_object_ori:
            line = line.lower()
            if patt_module.match(line):
                isModule = True
            if patt_subroutine.match(line):
                hasSubroutine = True

        # If we have a module, close the input and cycle to next file.
        if isModule and not hasSubroutine:
            file_object_ori.close()
            continue

        # open modified file in write mode
        file_object_mod = open(os.path.join(DIR_MOD,f), 'w')

        # Go back to the beginning
        file_object_ori.seek(0)
        for line in file_object_ori:
            # Just deal with lower case string
            line = line.lower()

            # Keep the differentiated routine for the solve_d command
            if 'subroutine' in line and 'end' not in line:
                if '_d' in line:
                    flag_d = True
                else:
                    flag_d = False

            if 'use' in line and 'only' in line:
                if flag_d:
                    line = line[:-1] + '_d' + '\n'

            # Replace _cd on calls
            if '_cd' in line:
                line = line.replace('_cd', '')

            # Replace _d modules with normal -- except for the useful
            # ones.
            m = patt_modules.match(line)
            if m:
                found = False
                for m in useful_modules:
                    if m in line:
                        found = True
                if not found:
                    line = line.replace('_d', '')

            if 'external solve' not in line:
                # write the modified line to new file
                file_object_mod.write(line)

        # close the files
        file_object_ori.close()
        file_object_mod.close()

        # success message
        print " Modified file saved", file_object_mod.name
