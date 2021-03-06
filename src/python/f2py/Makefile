#      ******************************************************************
#      *                                                                *
#      * File:          Makefile                                        *
#      * Authors: Ney Secco and John Jasa                               *
#      * Based on Gaetan Kenway's Makefiles                             *
#      * Starting date: 07-27-2016                                      *
#      * Last modified: 07-27-2016                                      *
#      *                                                                *
#      ******************************************************************

HOME_DIR = ../../..
COMMON_FILE = $(HOME_DIR)/common.mk
RULES_FILE  = $(HOME_DIR)/rulesSources.mk
include ${COMMON_FILE}
include ${RULES_FILE}

LINKER_ALL_FLAGS = -L$(LIBDIR) -loas $(CGNS_LINKER_FLAGS) \
		    $(PETSC_LINKER_FLAGS) $(LINKER_FLAGS) $(CGNS_LINKER_FLAGS)

OAS_API_OBJECTS = fortranobject.o \
	          OAS_APImodule.o \
	          OAS_API-f2pywrappers2.o \

# This file is sometimes for F77 wrapper
#		 hypsurf-f2pywrappers.o \

default: all

all:

# Generate Python inlude directory
	 $(eval PYTHON_INCLUDES = $(shell python-config --includes))
	 @echo "#------------------------------------------------------#"
	 @echo Python Include Flags $(PYTHON_INCLUDES)
	 @echo "#------------------------------------------------------#"

# Generate Numpy inlude directory
	$(eval NUMPY_INCLUDES = $(shell python -c 'import numpy; print numpy.get_include()'))
	 @echo "#------------------------------------------------------#"
	 @echo Numpy Include Directory: $(NUMPY_INCLUDES)
	 @echo "#------------------------------------------------------#"
# Generate f2py root directory
	$(eval F2PY_ROOT = $(shell python get_f2py.py))
	 @echo "#------------------------------------------------------#"
	 @echo f2py root directory: $(F2PY_ROOT)
	 @echo "#------------------------------------------------------#"

#       need to have f2py_f2cmap in this folder....
# This will link all realType, and intType to equivalent variables in C
	cp f2py_f2cmap.ref .f2py_f2cmap

# Run f2py on signature files only
# f2py will give us C wrappers that we will compile later on.
# We do this to avoid the default compilers used by f2py.
# In summary, f2py will not compile anything, it will just generate
# the C wrapper we will compile ourselves.
	f2py OAS_API.pyf
	@echo " "
	$(CC) $(CC_ALL_FLAGS) $(PYTHON_INCLUDES) -I$(NUMPY_INCLUDES) \
	-I$(F2PY_ROOT)/src -c OAS_APImodule.c

# Compile the fortran wrapper from f2py directory
	$(CC) $(CC_ALL_FLAGS) $(PYTHON_INCLUDES) -I$(NUMPY_INCLUDES) -c \
	$(F2PY_ROOT)/src/fortranobject.c -o fortranobject.o

# # This file is for F77 wrappers if they exist
#	$(FF90) $(FF90_ALL_FLAGS) -I$(MODDIR) -c OAS_API-f2pywrappers.f

# Compile fortran wrappers generated by f2py
	$(FF90) $(FF90_ALL_FLAGS) -I$(MODDIR) -c OAS_API-f2pywrappers2.f90

# Link everything to make python modules
	$(FF90) -shared $(OAS_API_OBJECTS) $(LINKER_ALL_FLAGS) -o OAS_API.so

# Test if modules can be imported
	python importTest.py

# Move python modules to the main python folder
	mv *.so ../../../python
# Clean up temporary files
	-rm *.o
	-rm *.f90 *.c
	-rm .f2py_f2cmap

clean:
	@echo "        Making clean in f2py... "
	rm -f $(MAKE_CLEAN_ARGUMENTS)

#src:   $(FF90_OBJECTS) $(CC_OBJECTS)
