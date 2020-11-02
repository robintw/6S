###############################################################################
#
# CMake build scripts for 6SV
# 
# Created 10/07/2014 by Pete Bunting (petebunting@mac.com)
#
###############################################################################

###############################################################################
# Set Project name and version
project (SIXSV)

# The version number.
set (SIXS_VERSION "1.1.1")

# CMake settings
cmake_minimum_required(VERSION 2.6.0)

enable_language (Fortran)

IF(NOT CMAKE_BUILD_TYPE)
  #SET(CMAKE_BUILD_TYPE "DEBUG")
  SET(CMAKE_BUILD_TYPE "RELEASE")
  #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
  #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

set(CMAKE_COLOR_MAKEFILE ON)

# Allow advanced users to generate Makefiles printing detailed commands
mark_as_advanced(CMAKE_VERBOSE_MAKEFILE)

# Path to additional CMake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ./cmake/modules/)

# Set up compiler flags
# FFLAGS depend on the compiler
get_filename_component (Fortran_COMPILER_NAME ${CMAKE_Fortran_COMPILER} NAME)

if (Fortran_COMPILER_NAME MATCHES "gfortran.*")
  # gfortran
  set (CMAKE_Fortran_FLAGS_RELEASE "-ffpe-summary='none' -funroll-all-loops -std=legacy -ffixed-line-length-none -fno-f2c -O3")
  set (CMAKE_Fortran_FLAGS_DEBUG   "-ffpe-summary='none' -fno-f2c -O0 -g -std=legacy -ffixed-line-length-none")
elseif (Fortran_COMPILER_NAME MATCHES "ifort.*")
  # ifort (untested)
  set (CMAKE_Fortran_FLAGS_RELEASE "-f77rtl -O3 -fPIC -static-intel ")
  set (CMAKE_Fortran_FLAGS_DEBUG   "-f77rtl -O0 -g -fPIC -static-intel ")
elseif (Fortran_COMPILER_NAME MATCHES "g77")
  # g77
  set (CMAKE_Fortran_FLAGS_RELEASE "-funroll-all-loops -fno-f2c -O3 -m32")
  set (CMAKE_Fortran_FLAGS_DEBUG   "-fno-f2c -O0 -g -m32")
else (Fortran_COMPILER_NAME MATCHES "gfortran.*")
  message ("CMAKE_Fortran_COMPILER full path: " ${CMAKE_Fortran_COMPILER})
  message ("Fortran compiler: " ${Fortran_COMPILER_NAME})
  message ("No optimized Fortran compiler flags are known, we just try -O2...")
  set (CMAKE_Fortran_FLAGS_RELEASE "-O2")
  set (CMAKE_Fortran_FLAGS_DEBUG   "-O0 -g")
endif (Fortran_COMPILER_NAME MATCHES "gfortran.*")


# BUILD EXE
add_executable(sixs ./src/main.f ./src/AATSR.f ./src/ABSTRA.f ./src/AEROPROF.f ./src/AEROSO.f ./src/AKTOOL.f ./src/ALI.f ./src/ASTER.f ./src/ATMREF.f ./src/AVHRR.f ./src/BBM.f ./src/BDM.f ./src/BRDFGRID.f ./src/CHAND.f ./src/CLEARW.f ./src/CSALBR.f ./src/DICA1.f ./src/DICA2.f ./src/DICA3.f ./src/DISCOM.f ./src/DISCRE.f ./src/DUST.f ./src/ENVIRO.f ./src/EQUIVWL.f ./src/ETM.f ./src/GAUSS.f ./src/GLI.f ./src/GOES.f ./src/HAPKALBE.f ./src/HAPKBRDF.f ./src/HRV.f ./src/HYPBLUE.f ./src/IAPIALBE.f ./src/IAPIBRDF.f ./src/IAPITOOLS.f ./src/INTERP.f ./src/ISO.f ./src/KERNEL.f ./src/KERNELPOL.f ./src/LAKEW.f ./src/MAS.f ./src/MERIS.f ./src/METEO.f ./src/METH1.f ./src/METH2.f ./src/METH3.f ./src/METH4.f ./src/METH5.f ./src/METH6.f ./src/MIDSUM.f ./src/MIDWIN.f ./src/MIE.f ./src/MINNALBE.f ./src/MINNBRDF.f ./src/MOCA1.f ./src/MOCA2.f ./src/MOCA3.f ./src/MOCA4.f ./src/MOCA5.f ./src/MOCA6.f ./src/MODIS.f ./src/MODISALBE.f ./src/MODISBRDF.f ./src/MSS.f ./src/NIOX1.f ./src/NIOX2.f ./src/NIOX3.f ./src/NIOX4.f ./src/NIOX5.f ./src/NIOX6.f ./src/OCEA.f ./src/OCEAALBE.f ./src/OCEABRDF.f ./src/OCEABRDFFAST.f ./src/OCEATOOLS.f ./src/ODA550.f ./src/ODRAYL.f ./src/OS.f ./src/OSPOL.f ./src/OXYG3.f ./src/OXYG4.f ./src/OXYG5.f ./src/OXYG6.f ./src/OZON1.f ./src/PLANPOL.f ./src/POLDER.f ./src/POLGLIT.f ./src/POLNAD.f ./src/POSGE.f ./src/POSGW.f ./src/POSLAN.f ./src/POSMTO.f ./src/POSNOA.f ./src/POSSOL.f ./src/POSSPO.f ./src/PRESPLANE.f ./src/PRESSURE.f ./src/PRINT_ERROR.f ./src/RAHMALBE.f ./src/RAHMBRDF.f ./src/ROUJALBE.f ./src/ROUJBRDF.f ./src/SAND.f ./src/SCATRA.f ./src/SEAWIFS.f ./src/SOLIRR.f ./src/SOOT.f ./src/SPECINTERP.f ./src/SPLIE2.f ./src/SPLIN2.f ./src/SPLINE.f ./src/SPLINT.f ./src/STM.f ./src/SUBSUM.f ./src/SUBWIN.f ./src/TM.f ./src/TROPIC.f ./src/TRUNCA.f ./src/US62.f ./src/VARSOL.f ./src/VEGETA.f ./src/VERSALBE.f ./src/VERSBRDF.f ./src/VERSTOOLS.f ./src/VGT.f ./src/VIIRS.f ./src/WALTALBE.f ./src/WALTBRDF.f ./src/WATE.f ./src/WAVA1.f ./src/WAVA2.f ./src/WAVA3.f ./src/WAVA4.f ./src/WAVA5.f ./src/WAVA6.f)

# INSTALL
install (TARGETS sixs DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)


