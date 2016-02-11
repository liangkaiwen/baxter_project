# Find the header files

FIND_PATH(G2O_INCLUDE_DIR g2o/core/base_vertex.h
  ${G2O_ROOT}/include
  $ENV{G2O_ROOT}/include
  $ENV{G2O_ROOT}
  /usr/local/include
  /usr/include
  /opt/local/include
  /sw/local/include
  /sw/include
  NO_DEFAULT_PATH
  )
  
# Peter adds
SET(G2O_INCLUDE_DIRS ${G2O_INCLUDE_DIR} ${G2O_INCLUDE_DIR}/EXTERNAL/csparse)

# Macro to unify finding both the debug and release versions of the
# libraries; this is adapted from the OpenSceneGraph FIND_LIBRARY
# macro.

MACRO(FIND_G2O_LIBRARY MYLIBRARY MYLIBRARYNAME)

  FIND_LIBRARY("${MYLIBRARY}_DEBUG"
    NAMES "g2o_${MYLIBRARYNAME}_d"
    PATHS
    ${G2O_ROOT}/lib/Debug
    ${G2O_ROOT}/lib
    $ENV{G2O_ROOT}/lib/Debug
    $ENV{G2O_ROOT}/lib
    NO_DEFAULT_PATH
    )

  FIND_LIBRARY("${MYLIBRARY}_DEBUG"
    NAMES "g2o_${MYLIBRARYNAME}_d"
    PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local/lib
    /usr/local/lib64
    /usr/lib
    /usr/lib64
    /opt/local/lib
    /sw/local/lib
    /sw/lib
    )
  
  FIND_LIBRARY(${MYLIBRARY}
    NAMES "g2o_${MYLIBRARYNAME}"
    PATHS
    ${G2O_ROOT}/lib/Release
    ${G2O_ROOT}/lib

    $ENV{G2O_ROOT}/lib/Release
    $ENV{G2O_ROOT}/lib
    NO_DEFAULT_PATH
    )

  FIND_LIBRARY(${MYLIBRARY}
    NAMES "g2o_${MYLIBRARYNAME}"
    PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local/lib
    /usr/local/lib64
    /usr/lib
    /usr/lib64
    /opt/local/lib
    /sw/local/lib
    /sw/lib
    )
  
  IF(NOT ${MYLIBRARY}_DEBUG)
    IF(MYLIBRARY)
      SET(${MYLIBRARY}_DEBUG ${MYLIBRARY})
    ENDIF(MYLIBRARY)
  ENDIF( NOT ${MYLIBRARY}_DEBUG)
    
ENDMACRO()

# Peter:
# MUCH smarter would be to use a macro or function to set both the variable and its debug equivalent

# Find the core elements
FIND_G2O_LIBRARY(G2O_STUFF_LIBRARY stuff)
FIND_G2O_LIBRARY(G2O_CORE_LIBRARY core)
SET(G2O_LIBRARIES ${G2O_LIBRARIES} ${G2O_STUFF_LIBRARY} ${G2O_CORE_LIBRARY})
SET(G2O_LIBRARIES_DEBUG ${G2O_LIBRARIES_DEBUG} ${G2O_STUFF_LIBRARY_DEBUG} ${G2O_CORE_LIBRARY_DEBUG})

# Find the CLI library
FIND_G2O_LIBRARY(G2O_CLI_LIBRARY cli)
SET(G2O_LIBRARIES ${G2O_LIBRARIES} ${G2O_CLI_LIBRARY})
SET(G2O_LIBRARIES_DEBUG ${G2O_LIBRARIES_DEBUG} ${G2O_CLI_LIBRARY_DEBUG})

# Find the pluggable solvers
# Peter mod: remove those I know I don't have so I can do a simple set on whole library list
# and so I dont have to actually check whether set
#FIND_G2O_LIBRARY(G2O_SOLVER_CHOLMOD solver_cholmod)
FIND_G2O_LIBRARY(G2O_SOLVER_CSPARSE solver_csparse)
FIND_G2O_LIBRARY(G2O_SOLVER_CSPARSE_EXTENSION csparse_extension)
FIND_G2O_LIBRARY(G2O_SOLVER_DENSE solver_dense)
FIND_G2O_LIBRARY(G2O_SOLVER_PCG solver_pcg)
FIND_G2O_LIBRARY(G2O_SOLVER_SLAM2D_LINEAR solver_slam2d_linear)
FIND_G2O_LIBRARY(G2O_SOLVER_STRUCTURE_ONLY solver_structure_only)
#FIND_G2O_LIBRARY(G2O_SOLVER_EIGEN solver_eigen)
SET(G2O_LIBRARIES ${G2O_LIBRARIES} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_CSPARSE_EXTENSION} ${G2O_SOLVER_DENSE} ${G2O_SOLVER_PCG} ${G2O_SOLVER_SLAM2D_LINEAR} ${G2O_SOLVER_STRUCTURE_ONLY})
SET(G2O_LIBRARIES_DEBUG ${G2O_LIBRARIES_DEBUG} ${G2O_SOLVER_CSPARSE_DEBUG} ${G2O_SOLVER_CSPARSE_EXTENSION_DEBUG} ${G2O_SOLVER_DENSE_DEBUG} ${G2O_SOLVER_PCG_DEBUG} ${G2O_SOLVER_SLAM2D_LINEAR_DEBUG} ${G2O_SOLVER_STRUCTURE_ONLY_DEBUG})

# Find the predefined types
FIND_G2O_LIBRARY(G2O_TYPES_DATA types_data)
FIND_G2O_LIBRARY(G2O_TYPES_ICP types_icp)
FIND_G2O_LIBRARY(G2O_TYPES_SBA types_sba)
FIND_G2O_LIBRARY(G2O_TYPES_SCLAM2D types_sclam2d)
FIND_G2O_LIBRARY(G2O_TYPES_SIM3 types_sim3)
FIND_G2O_LIBRARY(G2O_TYPES_SLAM2D types_slam2d)
FIND_G2O_LIBRARY(G2O_TYPES_SLAM3D types_slam3d)
SET(G2O_LIBRARIES ${G2O_LIBRARIES} ${G2O_TYPES_DATA} ${G2O_TYPES_ICP} ${G2O_TYPES_SBA} ${G2O_TYPES_SCLAM2D} ${G2O_TYPES_SIM3} ${G2O_TYPES_SLAM2D} ${G2O_TYPES_SLAM3D})
SET(G2O_LIBRARIES_DEBUG ${G2O_LIBRARIES_DEBUG} ${G2O_TYPES_DATA_DEBUG} ${G2O_TYPES_ICP_DEBUG} ${G2O_TYPES_SBA_DEBUG} ${G2O_TYPES_SCLAM2D_DEBUG} ${G2O_TYPES_SIM3_DEBUG} ${G2O_TYPES_SLAM2D_DEBUG} ${G2O_TYPES_SLAM3D_DEBUG})

# Peter adds additional libraries
if (MSVC)
# notice that these 3 lines go together:
FIND_G2O_LIBRARY(G2O_EXT_CSPARSE ext_csparse)
SET(G2O_LIBRARIES ${G2O_LIBRARIES} ${G2O_EXT_CSPARSE})
SET(G2O_LIBRARIES_DEBUG ${G2O_LIBRARIES_DEBUG} ${G2O_EXT_CSPARSE_DEBUG})
endif()
# These exist on windows, but aren't needed (yet)
#FIND_G2O_LIBRARY(G2O_EXT_FREEGLUT_MINIMAL ext_freeglut_minimal)
#FIND_G2O_LIBRARY(G2O_INTERFACE g2o_interface)
#FIND_G2O_LIBRARY(G2O_OPENGL_HELPER g2o_opengl_helper)


# G2O solvers declared found if we found at least one solver
SET(G2O_SOLVERS_FOUND "NO")
IF(G2O_SOLVER_CHOLMOD OR G2O_SOLVER_CSPARSE OR G2O_SOLVER_DENSE OR G2O_SOLVER_PCG OR G2O_SOLVER_SLAM2D_LINEAR OR G2O_SOLVER_STRUCTURE_ONLY OR G2O_SOLVER_EIGEN)
  SET(G2O_SOLVERS_FOUND "YES")
ENDIF(G2O_SOLVER_CHOLMOD OR G2O_SOLVER_CSPARSE OR G2O_SOLVER_DENSE OR G2O_SOLVER_PCG OR G2O_SOLVER_SLAM2D_LINEAR OR G2O_SOLVER_STRUCTURE_ONLY OR G2O_SOLVER_EIGEN)

# G2O itself declared found if we found the core libraries and at least one solver
SET(G2O_FOUND "NO")
IF(G2O_STUFF_LIBRARY AND G2O_CORE_LIBRARY AND G2O_INCLUDE_DIR AND G2O_SOLVERS_FOUND)
  SET(G2O_FOUND "YES")
ENDIF(G2O_STUFF_LIBRARY AND G2O_CORE_LIBRARY AND G2O_INCLUDE_DIR AND G2O_SOLVERS_FOUND)



