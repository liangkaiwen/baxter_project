/***********************************************************************
 *                        INTEL CONFIDENTIAL                           *
 ***********************************************************************

      Copyright 2010 Intel Corporation All Rights Reserved.
    The source code contained or described herein and all documents
    related to the source code ("Material") are owned by Intel
    Corporation or its suppliers or licensors. Title to the Material
    remains with Intel Corporation or its suppliers and licensors.
    The Material contains trade secrets and proprietary and
    confidential information of Intel or its suppliers and licensors.
    The Material is protected by worldwide copyright and trade secret
    laws and treaty provisions. No part of the Material may be used,
    copied, reproduced, modified, published, uploaded, posted,
    transmitted, distributed, or disclosed in any way without Intel's
    prior express written permission.

      No license under any patent, copyright, trade secret or other
    intellectual property right is granted to or conferred upon you by
    disclosure or delivery of the Materials, either expressly, by
    implication, inducement, estoppel or otherwise. Any license under
    such intellectual property rights must be express and approved by
    Intel in writing.

 **********************************************************************
    $File: $
    $Author:  $
    $Revision:  $
    $DateTime:  $
 **********************************************************************
    Description: Common includes for IOCTLS

    Other:

 **********************************************************************
	Supported format:

	Color Surfaces section
	======================
	YUY2							Color output in YUY2 format. Output as UINT16 per pixel
	RGB24							Color output in RGB24 format. Output as 3xUINT8 per pixel
	RGB32							Color output in RGB32 format. Output as 4xUINT8 per pixel

	Depth Surfaces section
	======================
	Z RAW IVCAM UINT16				Depth Z in IVCAM native 1/32mm unit. Output as UINT16 per pixel
	Z in MM in FLOAT32				Depth Z in milimeters. Output as float32 per pixel. Conversion done by casting from raw data integer to float32 and multiplying by 1/32.
	Z in MM in INT16				Depth Z in milimeters. Raw data rounded to MM.
	
	R RAW IVCAM UINT16				Depth R in IVCAM native 1/32mm unit. Output as UINT16 per pixel
	R in MM in FLOAT32				Depth R in milimeters. Output as float32 per pixel. Conversion done by casting from raw data integer to float32 and multiplying by 1/32.
	R in MM in INT16				Depth R in milimeters. Raw data rounded to MM.

	Vertices in MM in FLOAT32		X,Y,Z Vertices Output as float32 in milimeters. Get data directly from Unproject function (Calibration.h)
	Vertices in MM in INT16			X,Y,Z Vertices Output as INT16   in milimeters. Get data directly from Unproject function (Calibration.h)
	
	UVMap_Float32					0-1
	UVMap_UINT16					0- (65K as 1) - Get float32 from Unproject. Multiply by 2^16 - 1. Cast to UINT16. 

	Special formats for intarnal usage
	==================================
	CALIBRATION_40BIT				Calibration format.	Todo: add better description
	RAW_UINT16						GUID "CRAW"			Todo: add better description
	Z_I_NATIVE_24BIT				No IR synch			Todo: add better description


	Intensity Surfaces section
	IR_UINT16						Mainly for SDK Usage:
									8bpp data multiplied by 4 (Shift right 2 - two zero LSB) to keep 10bpp format as in CREATIVE.
									IR is SYNCHED with Depth. (IR_n is kept to be attached to Depth_n+1)
									output as UINT16
	
	IR_UINT8						Native IVCAM 8 bpp data.
									IR is SYNCHED with Depth. (IR_n is kept to be attached to Depth_n+1)
									output as UINT8

									Also used for:
									Stand alone format with no Depth image. high FPS.
									8BPP data
									No IR synch (No depth)
									Output as UINT8
									For B0 - Camera gives 2x12 bit per pixel with only 8 LSB relevant data
									For C0 - 8 bit per pixel

	IRPACKEDRAW_12BPPDATA_UINT16	Stand alone format with no Depth image.
									Camera gives 2x12 in each pixel. DLL gives 2xUINT16
									No IR synch (No depth)
											
 *********************************************************************/

#pragma once
#ifndef _IVCAM_TYPES_H
#define _IVCAM_TYPES_H


#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <string.h>
using namespace std;

#ifdef WIN32
# include <windows.h>
# define M_PI 3.14159265
#else
# define IN
# define OUT
# define MAX_PATH           PATH_MAX
 typedef unsigned long      DWORD;
 typedef short              WCHAR;
 typedef void *             HANDLE;
 typedef unsigned short     WORD;
 typedef unsigned int       BOOL;
 typedef long long          LONGLONG;
 typedef unsigned int       UINT;
 typedef unsigned char      byte;
 typedef unsigned char      BYTE;
 typedef char               INT8;
 typedef unsigned char      UINT8;
 typedef short              INT16;
 typedef unsigned short     UINT16;
 typedef long               INT32;
 typedef unsigned long      UINT32;
 typedef unsigned long long UINT64;
 typedef long long          INT64;

 typedef struct _SYSTEMTIME {
   WORD wYear;
   WORD wMonth;
   WORD wDayOfWeek;
   WORD wDay;
   WORD wHour;
   WORD wMinute;
   WORD wSecond;
   WORD wMilliseconds;
} SYSTEMTIME, *PSYSTEMTIME;

#define strcpy_s            strcpy
#define wcscpy_s            wcscpy
#define memcpy_s            memcpy
#define __stdcall

#endif


//

namespace Ivcam
{

	#define ivUINT8  unsigned byte 
	#define ivUINT16 unsigned short
	#define ivUINT32 unsigned long 
	#define ivUINT64 unsigned long long 

	#define NUM_OF_CALIBRATION_PARAMS   (100)
    #define NUM_OF_CALIBRATION_COEFFS   (100)

	#define HW_MONITOR_COMMAND_SIZE			  (1000)
	#define HW_MONITOR_BUFFER_SIZE               (1000)
	#define PARAMETERS_BUFFER_SIZE    (50)

	#define MAX_VENDOR_SPECIFIC_STRING_SIZE	60 

	#define REAL_VENDOR_SPECIFIC_STRING_SIZE_DPT	51
	#define REAL_VENDOR_SPECIFIC_STRING_SIZE_RGB	49 
	#define REAL_VENDOR_SPECIFIC_STRING_SIZE_DEV_NAME	45
	#define REAL_VENDOR_SPECIFIC_STRING_SIZE_HW_MONITOR	45 
	#define VENDOR_SPECIFIC_STRING_IS_VALID 1
	#define VENDOR_SPECIFIC_STRING_PADDING 0

	typedef int STREAM_HANDLE;
	typedef INT64 FRAME_HANDLE;

	typedef enum
	{
		LogLevel_Info		= 1		,
		LogLevel_Warning	= 10	,
		LogLevel_Error		= 50	,
		LogLevel_Nolog		= 1000	
	} TLogLevel;


	 /*! \enum TIVCAM_STATUS
	 *
	 */
	//! Defines status codes
	/** Driver API/Functions status/error reporting enumeration */
	typedef enum {
		ivcamSuccess										= 0, /*!< Operation success                                   */

		//====================================================================================================================
		ivcam_ERROR_START									= 1000,
		//====================================================================================================================
		ivcamFailure										= 1010, /*!< Operation failure                                   */
		ivcamNullArgument									= 1020, /*!< Received null argument                              */
		ivcamWrongArgument									= 1030, /*!< Received wrong argument                             */
		ivcamOutOfMemory									= 1040, /*!< Out of memory error                                 */
		ivcamValueNotValid									= 1050, /*!< Invalid value                                       */
		ivcamValueOutOfRange								= 1060, /*!< Value is out of range                               */
		ivcamTimeOut										= 1070, /*!< Time out                                            */
		ivcamNotImplemented									= 1080, /*!< Feature is not implemented                          */
		ivcamNotSupported  									= 1090, /*!< Feature is not supported                            */
		ivcamNotInitialized									= 1100, /*!< Feature not initialized                             */
		ivcamValueNotFound  								= 1110, /*!< Value not found                                     */
		ivcamAlreadyOpened									= 1120, /*!< Requested object already opened/exists              */
		ivcamDeviceNotConnected								= 1130, /*!< Requested device not connected                      */
		ivcamInvalidSyncScrope  							= 1140, /*!< Sync function called at async scope and vice versa  */
		ivcamDeviceNotConfigured							= 1150, 
		ivcamItemUnAvailable  								= 1160, 
		ivcamSupporedInInternalBuildOnly					= 1170, 
		ivcamFWRevisionNotSupported							= 1180, 
		ivcamBufferIsTooSmall								= 1190, 
		ivcamCanNotSetCurrentProfile						= 1210, 
		ivcamCanNotStartStream								= 1220, 
		ivcamFirmwareUpdateFailed							= 1230, 
		ivcamSecondStreamOfSameTypeNotAllowed				= 1240, 
		ivcamDFUFailedOEMIDMissMatch						= 1250, 
		ivcamObjectNotExist  								= 1260, 
		ivcamStreamOfSameTypeOlreadyExist  					= 1270, 
		ivcamCamInRecoveryFWUpdateRecommended 				= 1280, 
		ivcamCamMFCreateAttributesFailed 					= 1290,
		ivcamCamMFCreateDeviceSourceActivate  				= 1300, 
		ivcamCamcreatDevice  								= 1310, 
		ivcamCamActivateCurrentHwColorConfigurationFailed	= 1320,
		ivcamCannotGetWinUsbMutex							= 1330,
		ivcamStreamHandleDoesNotPointToAnyStream			= 1340,
		ivcamClosingInProgressRequestIgnored				= 1350,
		ivcamCannotUnregisterColorWhenSyncIsRegistered		= 1360,
		ivcamCannotUnregisterDepthWhenSyncIsRegistered		= 1370,
		//====================================================================================================================
		//DFU Section
		//====================================================================================================================
		ivcamInstallDriverFailed							= 2010, 
		ivcamDFUmodeDetectedWhileNotExpected				= 2020,
		ivcamDFUGetStatusFailed								= 2030,


		//====================================================================================================================
		//Calibration Section
		//====================================================================================================================
		ivcamOldCalibration									= 3010, 

		//====================================================================================================================
		//HWMonitor Section - general
		//====================================================================================================================
		ivcam_Firmware_ERROR_WrongCommand					= 4001,
		ivcam_Firmware_ERROR_StartNGEndAddr					= 4002,
		ivcam_Firmware_ERROR_AddressSpaceNotAligned			= 4003,
		ivcam_Firmware_ERROR_AddressSpaceTooSmall			= 4004,
		ivcam_Firmware_ERROR_MemoryAddrOverflow				= 4005,
		ivcam_Firmware_ERROR_WrongParameter					= 4006,
		ivcam_Firmware_ERROR_HWNotReady						= 4007,
		ivcam_Firmware_ERROR_I2CAccessFailed				= 4008,
		ivcam_Firmware_ERROR_NoExpectedUserAction			= 4009,
		ivcam_Firmware_ERROR_IntegrityError					= 4010,
		ivcam_Firmware_ERROR_NullOrZeroSizeString			= 4011,
		ivcam_Firmware_ERROR_GPIOPinNumberInvalid			= 4012,
		ivcam_Firmware_ERROR_GPIOPinDirectionInvalid		= 4013,
		ivcam_Firmware_ERROR_IlegalFlashAddress				= 4014,
		ivcam_Firmware_ERROR_IllegalUart2FlashSize			= 4015,
		ivcam_Firmware_ERROR_ParamsTableNotValid			= 4016,
		ivcam_Firmware_ERROR_ParamsTableIdNotValid			= 4017,
		ivcam_Firmware_ERROR_ParamsTableWrongExistingSize	= 4018,
		ivcam_Firmware_ERROR_WrongCRC						= 4019,
		ivcam_Firmware_ERROR_NotAuthorisedFlashWrite		= 4020,
		ivcam_Firmware_ERROR_NoDataToReturn					= 4021,
		ivcam_Firmware_ERROR_SpiReadFailed					= 4022,
		ivcam_Firmware_ERROR_SpiWriteFailed					= 4023,
		ivcam_Firmware_ERROR_SpiEraseSectorFailed			= 4024,
		ivcam_Firmware_ERROR_FlashTableisEmpty				= 4025,
		ivcam_Firmware_ERROR_LiguriaHumidityError			= 4026,
		ivcam_Firmware_ERROR_OacDisabled					= 4027,
		ivcam_Firmware_ERROR_I2cSeqDelay					= 4028,
		ivcam_Firmware_ERROR_CommandIsLocked				= 4029,
		ivcam_Firmware_ERROR_InvalidEeprom					= 4030,
		ivcam_Firmware_ERROR_ValueOutOfRange				= 4031,
		ivcam_Firmware_ERROR_InvalidDepthFormat				= 4032,
		ivcam_Firmware_ERROR_DepthFlowError					= 4033,
		ivcam_Firmware_ERROR_PmbEepromVersionNotSupported	= 4034,
		ivcam_Firmware_ERROR_PmbEepromIsEmpty				= 4035,
		ivcam_Firmware_ERROR_Timeout						= 4036,
		ivcam_Firmware_ERROR_ASICConfig						= 4037,
		ivcam_Firmware_ERROR_ASICPI							= 4038,
		ivcam_Firmware_ERROR_NotSafeCheckFailed				= 4039,
		ivcam_Firmware_ERROR_FlashRegionIsLocked			= 4040,

		//====================================================================================================================
		//ACPI Section -  
		//====================================================================================================================

		ivcam_Acpi_ERROR_ACPIDeviceNotFound					= 4200,
		ivcam_Acpi_ERROR_SetPowerModeFailed                 = 4210,
		ivcam_Acpi_ERROR_SetDFUModeFailed                   = 4220,

		//====================================================================================================================
		//HWMonitor Section - get last error
		//====================================================================================================================
		ivcam_Firmware_ERROR_etACTIVE						= 4500,
		
		// Eye safety errors (payload0)
		ivcam_Firmware_ERROR_etMSAFE_S1_ERR					= 4501,
		ivcam_Firmware_ERROR_etI2C_SAFE_ERR					= 4502,
		ivcam_Firmware_ERROR_etFLASH_SAFE_ERR				= 4503,
		
		// UVC errors
		ivcam_Firmware_ERROR_etI2C_CFG_ERR					= 4504,
		ivcam_Firmware_ERROR_etI2C_EV_ERR					= 4505,
		ivcam_Firmware_ERROR_etHUMIDITY_ERR					= 4506,
		ivcam_Firmware_ERROR_etMSAFE_S0_ERR					= 4507,
		ivcam_Firmware_ERROR_etLD_ERR						= 4508,
		ivcam_Firmware_ERROR_etPI_ERR						= 4509,
		ivcam_Firmware_ERROR_etPJCLK_ERR					= 4510,
		ivcam_Firmware_ERROR_END							= 4511,
		
		ivcam_ERROR_END										= 4999,
		//====================================================================================================================
		ivcam_WARNING_START									= 5000,
		//====================================================================================================================
		ivcamCameraAlreadyInitialized						= 5010, 
		ivcamAlreadyExist									= 5020,	
		ivcam_WARNING_END									= 9999,

		Last  											 
	} TIVCAM_STATUS;


	// Utility macros



	#define STRINGIFY(x) #x
	#define TOSTRING(x) STRINGIFY(x) 
	#define AT __FILE__ ":" TOSTRING(__LINE__)
	#define TEST __FILE__
	/*!
	   Checks \a Status for failure.
	   Logging the file and line location
	   and returning the error.
	*/

#ifdef WIN32
#define RETURN_ON_IVCAM_ERROR_CONTINUE_IF_WARNING(Status) if(((TIVCAM_STATUS)(Status) > ivcam_ERROR_START) &&  ((TIVCAM_STATUS)(Status) > ivcam_ERROR_END))	\
												{																			\
													LogService * TheLog = LogService::GetInstance();						\
													string str = AT;														\
													str += " - Failure at this line.";										\
													TheLog->Log(LogLevel_Error,str);										\
													return Status;															\
												}

#define RETURN_ON_IVCAM_FAILURE(Status)		if (((TIVCAM_STATUS)(Status)) != ivcamSuccess)								\
												{																			\
													LogService * TheLog = LogService::GetInstance();						\
													string str = AT;														\
													str += " - Failure at this line.";										\
													TheLog->Log(LogLevel_Error,str);										\
													return Status;															\
												}
	
#define LOG_IVCAM_FAILURE(Status)		if (((TIVCAM_STATUS)(Status)) != ivcamSuccess)								\
													{																			\
														LogService * TheLog = LogService::GetInstance();						\
														string str = AT;														\
														str += " - Failure at this line.";										\
														TheLog->Log(LogLevel_Error,str);										\
														}

#else
#define RETURN_ON_IVCAM_ERROR_CONTINUE_IF_WARNING(Status) if(((TIVCAM_STATUS)(Status) > ivcam_ERROR_START) &&  ((TIVCAM_STATUS)(Status) > ivcam_ERROR_END)) \
                                                {                                                                           \
                                                    return Status;                                                          \
                                                }

#define RETURN_ON_IVCAM_FAILURE(Status)     if (((TIVCAM_STATUS)(Status)) != ivcamSuccess)                              \
                                                {                                                                           \
                                                    return Status;                                                          \
                                                }

#define LOG_IVCAM_FAILURE(Status)       if (((TIVCAM_STATUS)(Status)) != ivcamSuccess)                              \
                                                    {                                                                           \
                                                        }
#endif
	
	/*!
	   Checks \a Status for failure.
	*/
	#define IVCAM_FAILURE(Status) (((TIVCAM_STATUS)(Status)) != ivcamSuccess)
	/*!
	   Checks \a Status for success.
	*/
	#define IVCAM_SUCCESS(Status) (((TIVCAM_STATUS)(Status)) == ivcamSuccess)



	typedef enum {	


		//QS
		HWType_QS =  1,
		HWType_QS1 = 2,
		HWType_QS2 = 3,
		HWType_QS3 = 4,
		HWType_QS4 = 5,
		HWType_QS5 = 6,
		HWType_QS6 = 7,
		HWType_QS7 = 8,
		HWType_QS8 = 9,
		HWType_QS9 = 100,
		HWType_QS10 = 101,
		HWType_QS11 = 102,
		HWType_QS12 = 103,
		HWType_QS13 = 104,
		HWType_QS14 = 105,
		HWType_QS15 = 106,
		HWType_QS16 = 107,

		//QSP
		HWType_QSP = 10,
		HWType_QSP1 = 11,
		HWType_QSP2 = 12,
		HWType_QSP3 = 13,
		HWType_QSP4 = 14,
		HWType_QSP5 = 15,
		HWType_QSP6 = 16,
		HWType_QSP7 = 17,
		HWType_QSP8 = 18,
		HWType_QSP9 = 200,
		HWType_QSP10 = 201,
		HWType_QSP11 = 202,
		HWType_QSP12 = 203,
		HWType_QSP13 = 204,
		HWType_QSP14 = 205,
		HWType_QSP15 = 206,
		HWType_QSP16 = 207,

		//PRQ
		HWType_PRQ = 19,

		//PRQ
		HWType_PRQ14A = 20,
		HWType_PRQ14B = 21,
		HWType_PRQ14C = 22,
		HWType_PRQ15A = 23,
		HWType_PRQ15B = 24,
		HWType_PRQ15C = 25,

		HWType_Last = 1000,
		//Not Available 
		HWType_None = 0xFFFFFFFF,

	} THWType;



	typedef struct HWRevisionInfo 
	{
		THWType HWType; 
		UINT32 OEMId;
		UINT32 AsicVer;
	}THWRevisionInfo;      


	typedef struct FWRevisionInfo 
	{
		int Major; 
		int Minor; 
		INT64 SerialNumber;
		byte RawData[256];
		int size;
		int AsicVersion;
		int FixNum;
		int BuildNum;
	}TFWRevisionInfo;      

	typedef struct DLLRevisionInfo 
	{
		int Major; 
		int Minor; 
		int FixNumber; 
		int BuildId;


		/*OBSOLETE*/int RevId; 
		/*OBSOLETE*/int ChnageSet; 
	}TDLLRevisionInfo;      

	typedef struct 
	{
		double data[NUM_OF_CALIBRATION_PARAMS];     
	}TCalibrationData;


	typedef struct 
	{
		float	Rmax;
		float	Kc[3][3];		//[3x3]: intrinsic calibration matrix of the IR camera
		float	Distc[5];		// [1x5]: forward distortion parameters of the IR camera
		float	Invdistc[5];	// [1x5]: the inverse distortion parameters of the IR camera
		float	Pp[3][4];		// [3x4] : projection matrix
		float	Kp[3][3];		// [3x3]: intrinsic calibration matrix of the projector
		float	Rp[3][3];		// [3x3]: extrinsic calibration matrix of the projector
		float	Tp[3];			// [1x3]: translation vector of the projector
		float	Distp[5];		// [1x5]: forward distortion parameters of the projector
		float	Invdistp[5];	// [1x5]: inverse distortion parameters of the projector
		float	Pt[3][4];		// [3x4]: IR to RGB (texture mapping) image transformation matrix
		float	Kt[3][3];
		float	Rt[3][3];
		float	Tt[3];
		float	Distt[5];		// [1x5]: The inverse distortion parameters of the RGB camera
		float	Invdistt[5];
		float   QV[6];
	}TCalibrationParameters;
	 

	typedef struct CalibrationDataWithVersion
	{
		int uniqueNumber; //Should be 0xCAFECAFE in Calibration version 1 or later. In calibration version 0 this is zero.
		int16_t TableValidation;
		int16_t TableVarsion;
		Ivcam::TCalibrationParameters CalibrationParameters;
	}TCalibrationDataWithVersion;
       
        // Maximum Z range (in cm or mm). [0,65535] in the Z image is mapped to [0,Zmax]


	typedef enum 
	{
		//--------------------part A - MF regular------------------------
		/** Non MF extension - MF regular */
		/** Single value properties */
		IVCAM_PROPERTY_COLOR_EXPOSURE					=	1,
		IVCAM_PROPERTY_COLOR_BRIGHTNESS					=   2,
		IVCAM_PROPERTY_COLOR_CONTRAST					=   3,
		IVCAM_PROPERTY_COLOR_SATURATION					=   4,
		IVCAM_PROPERTY_COLOR_HUE						=   5,
		IVCAM_PROPERTY_COLOR_GAMMA						=   6,
		IVCAM_PROPERTY_COLOR_WHITE_BALANCE				=   7,
		IVCAM_PROPERTY_COLOR_SHARPNESS					=   8,
		IVCAM_PROPERTY_COLOR_BACK_LIGHT_COMPENSATION	=   9,
		IVCAM_PROPERTY_COLOR_GAIN						=   10,
		IVCAM_PROPERTY_COLOR_POWER_LINE_FREQUENCY		=   11,
		IVCAM_PROPERTY_AUDIO_MIX_LEVEL					=   12,
		IVCAM_PROPERTY_APERTURE							=   13,
		//until here MF regular properties


		//-----------------------part B - internal -------------------------
		//**non MF properties*/
		//IVCAM_PROPERTY_DEPTH_SATURATION_VALUE			=   200,
		//IVCAM_PROPERTY_DEPTH_SMOOTHING				=   201,
		//IVCAM_PROPERTY_CAMERA_MODEL					=   206,

		IVCAM_PROPERTY_DISTORTION_CORRECTION_I			=	202 ,
		IVCAM_PROPERTY_DISTORTION_CORRECTION_DPTH		=	203 ,
		IVCAM_PROPERTY_DEPTH_MIRROR						=	204 , //0 - not mirrored, 1 - mirrored
		IVCAM_PROPERTY_COLOR_MIRROR						=	205 ,
    


		//--------------------------part C - extracted from calibration--------------------------
		//** Two value properties - extracted from calibration */
		IVCAM_PROPERTY_COLOR_FIELD_OF_VIEW				=   207,
		IVCAM_PROPERTY_COLOR_SENSOR_RANGE				=   209,
		IVCAM_PROPERTY_COLOR_FOCAL_LENGTH				=   211,
		IVCAM_PROPERTY_COLOR_PRINCIPAL_POINT			=   213,

		IVCAM_PROPERTY_DEPTH_FIELD_OF_VIEW				=   215,
		IVCAM_PROPERTY_DEPTH_UNDISTORTED_FIELD_OF_VIEW	=   223,
		IVCAM_PROPERTY_DEPTH_SENSOR_RANGE				=   217,
		IVCAM_PROPERTY_DEPTH_FOCAL_LENGTH				=   219,
		IVCAM_PROPERTY_DEPTH_UNDISTORTED_FOCAL_LENGTH	=   225,
		IVCAM_PROPERTY_DEPTH_PRINCIPAL_POINT			=   221,

          
 
		//--------------------------part D - MF extention -------------------------
		/** MF extention */
		IVCAM_PROPERTY_MF_DEPTH_LOW_CONFIDENCE_VALUE	=   5000,
		IVCAM_PROPERTY_MF_DEPTH_UNIT					=   5001,   // in micron
		/*IVCAM_PROPERTY_MF_FIRMWARE_UPDATE_MODE			=   5002,*/
		IVCAM_PROPERTY_MF_CALIBRATION_DATA				=   5003, 
		IVCAM_PROPERTY_MF_LASER_POWER					=   5004,
		IVCAM_PROPERTY_MF_ACCURACY						=   5005,
		IVCAM_PROPERTY_MF_INTENSITY_IMAGE_TYPE			=   5006 , //0 - (I0 - laser off), 1 - (I1 - Laser on), 2 - (I1-I0), default is I1.
		IVCAM_PROPERTY_MF_MOTION_VS_RANGE_TRADE			=   5007 ,
		IVCAM_PROPERTY_MF_POWER_GEAR					=   5008 ,
		IVCAM_PROPERTY_MF_FILTER_OPTION					=   5009 ,
		IVCAM_PROPERTY_MF_VERSION						=   5010 ,
		IVCAM_PROPERTY_MF_DEPTH_CONFIDENCE_THRESHOLD	=   5013,
	

		//---------------------------part E- internal----------------------------
		/** Misc. */
		IVCAM_PROPERTY_ACCELEROMETER_READING			=   3000,   // three values
		IVCAM_PROPERTY_PROJECTION_SERIALIZABLE			=   3003,	
		IVCAM_PROPERTY_CUSTOMIZED						=   0x04000000,
	}Property;



	struct ivcam_properties
	{
		int label;
		float defaultValue;
		float min;
		float max;
		float step;
		bool isAutoCap;
		int MFExtEU;
		int DataLen;
	};

	/** A structure for representing a range defined with float32 values */
	struct RangeF32 {
		float min, max;
	};


	typedef enum
	{
		Res_None			= 0	,	/*!< width: 320    height: 180      */
		Res_320x180				,	/*!< width: 320    height: 180      */
		Res_320x240				,	/*!< width: 320    height: 240      */
		Res_424x240				,	/*!< width: 424    height: 240      */
		Res_640x240				,	/*!< width: 640    height: 240      */	
		Res_640x480				,	/*!< width: 640    height: 480      */
		Res_640x360				,	/*!< width: 640    height: 360      */
		Res_848x480				,	/*!< width: 848    height: 480      */
		Res_960x540				,	/*!< width: 960    height: 540      */
		Res_1280x720			,	/*!< width: 1280   height: 720      */
		Res_1920x1080			,	/*!< width: 1920   height: 1080     */
		Res_Count
	} TSurfaceResolution;


	typedef enum
	{
		StreamType_Color,	
		StreamType_Depth,
		StreamType_Sync
	} TStreamType;


	typedef enum
	{
		ColorFormat_None = 0,
		ColorFormat_YUY2	,	
		ColorFormat_RGB24	,	
		ColorFormat_RGB32	,
		ColorFormat_INV_RAW16
	} TColorFormat;

	typedef enum
	{
		DepthFormat_None				  = 0,
	
		DepthFormat_Z					  = 1,//Depth	
		DepthFormat_R					  = 2,//Range	
		DepthFormat_HighFPSIntensity8bpp  = 4,//IR image, 300 FPS, 8BPP ( W * H * 1 ) in Plane[0]

		DepthFormat_Last				  = 5000
	} TDepthFormat;


	typedef enum
	{
		DepthOutput_UINT16_RAW_IVCAM	,	//UINT16 in units of 1/32 mm
		DepthOutput_FLOAT32_MM				//Float32 in units of 1 mm
	} TDepthOutputUnit;

	typedef enum
	{
		UVMap_None			=0	,
		/*OBSOLETE - DO NOT USE*/UVMap_INT16_0to65K		,	//INT16 ranging from 0 to 65K
		UVMap_FLOAT32_0to1			//Float32 ranging from 0 to 1
	} TUVMappingType;

	typedef enum
	{
		Vertices_None	=0,
		Vertices_XYZ_FLOAT32_MM,		//X,Y,Z vertices in float32 in mm
	} TVerticesType;

	typedef enum
	{
		RightHandedCoordinateSystem  =0,
		LeftHandedCoordinateSystem   
	} TCoordinatSystemDirection;

	typedef enum
	{
		IntensityAddition_None		=0	,
		IntensityAddition_8BPP_UINT8	,	//Raw IVCAM 8bpp in uint8
		IntensityAddition_10BPP_UINT16	,	//10bpp in uint16 (Actually it is 8bpp real data shifted right 2 to match older 10bpp formats.)
	} TIntensityAdditionType;

    /** A type representing a three-dimensional point defined with pxcF32 values */
    struct Point3DF32 {
        float x, y, z;
    };

    /** A type representing a two-dimensional point defined with pxcF32 values */
    struct Point2DF32 {
        float x, y;
    };

    typedef struct IRThermalLoopParams
    {
        IRThermalLoopParams(){
                IRThermalLoopEnable = 1;
                TimeOutA             = 10000;
                TimeOutB             = 0;
                TimeOutC             = 0;
                TransitionTemp   = 3;
                TempThreshold        = 2;
                HFOVsensitivity  = 0.025f;
                FcxSlopeA            = -0.003696988f;
                FcxSlopeB            = 0.005809239f;
                FcxSlopeC            = 0;
                FcxOffset            = 0;
                UxSlopeA             = -0.000210918f;
                UxSlopeB             = 0.000034253955f;
                UxSlopeC             = 0;
                UxOffset             = 0;
                LiguriaTempWeight    = 1;
                IrTempWeight         = 0;
                AmbientTempWeight    = 0;
                Param1           = 0;
                Param2           = 0;
                Param3           = 0;
                Param4           = 0;
                Param5           = 0;
        }
        float IRThermalLoopEnable;//enable the mechanism
        float TimeOutA;//default time out
        float TimeOutB;//reserved
        float TimeOutC;//reserved
        float TransitionTemp;//celcius degrees, the transition temperatures to ignore and use offset;
        float TempThreshold;//celcius degrees, the temperatures delta that above should be fixed;
        float HFOVsensitivity;
        float FcxSlopeA;// the temperature model fc slope a from slope_hfcx = ref_fcx*a + b
        float FcxSlopeB;// the temperature model fc slope b from slope_hfcx = ref_fcx*a + b
        float FcxSlopeC;//reserved
        float FcxOffset;// the temperature model fc offset
        float UxSlopeA;// the temperature model ux slope a from slope_ux = ref_ux*a + ref_fcx*b
        float UxSlopeB;// the temperature model ux slope b from slope_ux = ref_ux*a + ref_fcx*b
        float UxSlopeC;//reserved
        float UxOffset;// the temperature model ux offset
        float LiguriaTempWeight;// the liguria temperature weight in the temperature delta calculations
        float IrTempWeight;// the Ir temperature weight in the temperature delta calculations
        float AmbientTempWeight;//reserved
        float Param1;//reserved
        float Param2;//reserved
        float Param3;//reserved
        float Param4;//reserved
        float Param5;//reserved

    }TIRThermalLoopParams;


    typedef struct TemperatureData
    {
        float LiguriaTemp;
        float IRTemp;
        float AmbientTemp;
    }TTemperatureData;

	class TIVCAMStreamProfile 
	{
	public:
		TIVCAMStreamProfile(){};
	public:
		TStreamType StreamType;

		TSurfaceResolution		DepthResolution;
		TSurfaceResolution		ColorResolution;
		INT32					FPS;
		int						NumberOfImagesToHold;	//Number of images the application will hold and not release.

		TDepthFormat			DepthFormat; 
		TIntensityAdditionType	IntensityAdditionType;	
		TUVMappingType			UVMappingType;
		TVerticesType			VerticesType;
		TDepthOutputUnit		DepthOutputUnit;

		TColorFormat			ColorFormat;

		//Caller may put a pointer to additional data here
		void * UserContext;
	};

	typedef struct 
	{
		TStreamType				StreamType;
		TSurfaceResolution		Resolution;
		TDepthFormat			DepthFormat;
		bool					IntensityAddition;
		TColorFormat			ColorFormat;
		INT32					FPS;
	}TIVCAMCameraProfile;

	enum CameraModel 
	{
		CAMERA_MODEL_GENERIC    = 0x00000000,
		CAMERA_MODEL_DS325      = 0x00100245,
		CAMERA_MODEL_IVCAM      = 0x0020000E,
	};


	class TIVCAMImageHeader 
	{
	public:
		INT32 SizeOfHeader;
		INT32 SizeOfImageWithoutHeader;
		int Width;
		int Height;

		TIVCAMImageHeader(){SizeOfHeader = sizeof(TIVCAMImageHeader);};

	};



	class TIVCAMStreamWrapper 
     {
     public:
          TIVCAMStreamProfile StreamProfile;
          STREAM_HANDLE      StreamHandle;
          FRAME_HANDLE DepthHandleForRelease;
          FRAME_HANDLE ColorHandleForRelease;

 
          INT64 GetFrameNumber(){return FrameNumber;};
          void SetFrameNumber(INT64 FrameNum){FrameNumber = FrameNum;};

          /*OBSOLETE*/INT64 GetTimeStampInTicks(){return TimeStampInTicks;};
          /*OBSOLETE*/void SetTimeStampInTicks(INT64 TimeStamp){ TimeStampInTicks = TimeStamp;};


          INT64 GetColorTimeStampInTicks(){return ColorTimeStampInTicks;};
          void SetColorTimeStampInTicks(INT64 TimeStamp){ ColorTimeStampInTicks = TimeStamp;};

          INT64 GetDepthTimeStampInTicks(){return DepthTimeStampInTicks;};
          void SetDepthTimeStampInTicks(INT64 TimeStamp){ DepthTimeStampInTicks = TimeStamp;};

          INT64 GetStreamHandle(){return StreamHandle;};


          //** Depth section
          void* GetPointerToDepthPlane(){return DepthPlanes[0];};
          void SetPointerToDepthPlane(void* data){DepthPlanes[0] = data;};
          void* GetPointerToDepthPlaneWithHeader(){return DepthPlaneWithHeader;};
          void SetPointerToDepthPlaneWithHeader(void* data){DepthPlaneWithHeader = data;};

          void* GetPointerToIntensityPlane(){return DepthPlanes[1];};
          void SetPointerToIntensityPlane(void* data){DepthPlanes[1] = data;};

          void* GetPointerToUVMapPlane(){return DepthPlanes[2];};
          void SetPointerToUVMapPlane(void* data){DepthPlanes[2] = data;};

          void* GetPointerToVerticesPlane(){return DepthPlanes[3];};
          void SetPointerToVerticesPlane(void* data){DepthPlanes[3] = data;};

          INT32 GetDepthPlaneSize(){return DepthPlanesSizeInBytes[0];};
          INT32 GetIntensityPlaneSize(){return DepthPlanesSizeInBytes[1];};
          INT32 GetUVMapPlaneSize(){return DepthPlanesSizeInBytes[2];};
          INT32 GetVerticesPlaneSize(){return DepthPlanesSizeInBytes[3];};

 
          //**Color section
          byte* GetPointerToColorPlane(){return ColorPlane;};
          byte* GetPointerToColorPlaneWithHeader(){return ColorPlaneWithHeader;};
          void SetPointerToColorPlane(byte* data){ColorPlane = data;};
          void SetPointerToColorPlaneWithHeader(byte* data){ColorPlaneWithHeader = data;};

          INT32 GetColorPlaneSize(){return ColorPlaneSizeInBytes;};

          INT32 DepthPlanesSizeInBytes[4];
          INT32 ColorPlaneSizeInBytes;

		//This section will be private in next revision. Please avoid accessing directly !

		//private:

		//Depth Section
          void* DepthPlanes[4];
          void* DepthPlaneWithHeader;

          //Color Section
          byte* ColorPlane;
          byte* ColorPlaneWithHeader;

 
          INT64 FrameNumber;

 
          /*OBSOLETE*/
          //In Color stream this is the color image timestamp 
          //In Depth stream this is the Depth image timestamp 
          //In Sync stream this is the Depth image timestamp 
          /*OBSOLETE*/INT64 TimeStampInTicks;

 
          INT64 DepthTimeStampInTicks;
          INT64 ColorTimeStampInTicks;

          int  LostFramesBetweenLastTwoFrames;

          SYSTEMTIME SystemTime;
     };





	typedef struct DefaultControlsConfiguration 
	{
		UINT32 data1; 
		UINT32 data2; 
		UINT32 data3; 
	}TDefaultControlsConfiguration;      

	typedef enum
	{
		DeviceConnected						= 0	,	
		DeviceDisconnected					= 1	,
		FirmwareUpdateRecommended			= 2 ,
		CalibrationTableUpdateRecommended	= 3 ,
		Event_Count             	
	} TIVCAMEventType;

	class DeviceInfo
	{
	public:
		wchar_t         name[256];      /* device name */
		wchar_t         did[256];       /* device identifier */

		const DeviceInfo &operator=(const DeviceInfo &old)
		{
			wcscpy_s(did, old.did);
			wcscpy_s(name, old.name);	

			return *this;
		}
	};


	class TIVCAMDeviceDetails 
	{
	public:
		int DeviceIndex;
		DeviceInfo DevInfoColor;
		DeviceInfo DevInfoDepth;
		DeviceInfo DevInfoHWmonitor;
		char DeviceID[256];
		char DeviceLocation[256];
		bool RGBconnected;
		bool DPTconnected;
		bool HWmonitorConnected;


		TIVCAMDeviceDetails():
			RGBconnected(false),
			DPTconnected(false),
			HWmonitorConnected(false){}

		const TIVCAMDeviceDetails &operator=(const TIVCAMDeviceDetails &old)
		{
			DeviceIndex = old.DeviceIndex;
			DevInfoColor = old.DevInfoColor;
			DevInfoDepth = old.DevInfoDepth;	
			DevInfoHWmonitor = old.DevInfoHWmonitor;
			strcpy_s(DeviceID , old.DeviceID);
			strcpy_s(DeviceLocation , old.DeviceLocation);
			RGBconnected = old.RGBconnected;
		    DPTconnected = old.DPTconnected;
		    HWmonitorConnected = old.HWmonitorConnected;

			return *this;
		}
	};

	typedef struct 
	{
		TIVCAMEventType					EventType;		// connect / disconnect
		TIVCAMDeviceDetails				DeviceDetails;  
		/*OBSOLETE*/TIVCAMDeviceDetails* 			IVCAMDeviceConnectedList;  
		/*OBSOLETE*/int								NumOfIVCAMDeviceConnected;
	}TIVCAMEventDetails;

	typedef struct 
	{
		IN bool oneDirection;
		IN char sendCommandData[HW_MONITOR_COMMAND_SIZE];
		IN int sizeOfSendCommandData;
		IN long TimeOut;
		OUT char recievedOPcode[4];
		OUT char recievedCommandData[HW_MONITOR_BUFFER_SIZE];
		OUT int sizeOfRecievedCommandData;
	}TIVCAMCommandDetails;
	
	/*!
	   Callback functions definitions.
	*/
	typedef void (__stdcall* ImageCallback)(TIVCAMStreamWrapper * stream);  
	
	//Callback function for receiving events
	typedef void (__stdcall* EventCallback)(TIVCAMEventDetails* EventDetails );  //To do: add event types.
	
	
	//Callback function for getting status of Firmware update
	typedef void (__stdcall* FirmwareUpdateProgressCallBack)(int NumBytesWritten,int NumBytesTotal);



	typedef struct DfuStatus
	{
		UINT32                   OEM_Id;
		UINT16                   FW_lastVersion;
		UINT16                   FW_highestVersion;
		UINT16                   FW_DownloadCompleted;
		UINT16                   DFU_isLocked;
		UINT16                   DFU_version;
		unsigned char            moduleSerial[8];
		UINT32					hwType;
		unsigned char			hwType_valid;
		unsigned char			OEM_Id_valid;
		unsigned char           spare[36];

	}TDfuStatus;


		typedef struct OACOffsetData
	{
		int OACOffset1;
		int OACOffset2;
		int OACOffset3;
		int OACOffset4;
	}TOACOffsetData;

	//==========================================================
	//vendors controls definitions
	//OEMs may get\set default camera behavior with these parameters.
	//==========================================================
	typedef struct
	{
		bool	 COLOR_EXPOSURE_AUTO;	
		int      COLOR_EXPOSURE; 
		uint32_t COLOR_BACK_LIGHT_COMPENSATION;      
		uint32_t COLOR_BRIGHTNESS;         
		uint32_t COLOR_CONTRAST;           
		uint32_t COLOR_GAIN;           
		uint32_t COLOR_POWER_LINE_FREQUENCY;          
		uint32_t COLOR_HUE;        
		uint32_t COLOR_SATURATION;           
		uint32_t COLOR_SHARPNESS;     
		uint32_t COLOR_GAMMA;  
		bool	 COLOR_WHITE_BALANCE_AUTO;
		uint32_t COLOR_WHITE_BALANCE;  // copy from FW if outo         
	}
	RGBControlSet;

	typedef struct
	{	
		uint32_t DEPTH_FIRMWARE_UPDATE_MODE;   
		uint32_t DEPTH_CALIBRATION_DATA;  
		uint32_t DEPTH_LASER_POWER;   
		uint32_t DEPTH_ACCURACY;        
		uint32_t DEPTH_TIMESTAMP_ENABLE;        
		uint32_t DEPTH_LOW_CONFIDENCE_VALUE;        	           				
		uint32_t DEPTH_MOTION_VS_RANGE_TRADE;           
		uint32_t DEPTH_POWER_GEAR;                        
		uint32_t DEPTH_FILTER_OPTION;                     
		uint32_t DEPTH_VERSION;                           
		uint32_t DEPTH_CONFIDENCE_THRESHOLD;             
	}
	DPTControlSet;

	typedef struct OEMInfo 
	{
		LONGLONG OemId; 
		uint16_t m_READ_ONLY_VendorId; // = 0x8086,
		uint16_t m_READ_ONLY_DeviceId; // = 0x0A66,
		char m_READ_ONLY_ControlDescriptorDPT[MAX_VENDOR_SPECIFIC_STRING_SIZE]; // = "DPT",
		char m_READ_ONLY_ControlDescriptorRGB[MAX_VENDOR_SPECIFIC_STRING_SIZE]; // = "RGB",
		char m_READ_ONLY_ControlDescriptorDevName[MAX_VENDOR_SPECIFIC_STRING_SIZE]; // = "IVCAM",
		char m_READ_ONLY_ControlDescriptorHWM[MAX_VENDOR_SPECIFIC_STRING_SIZE]; // = "HW_MONITOR",

		RGBControlSet RGB_ControlSet;
		DPTControlSet DPT_ControlSet;
	}TOEMInfo;      

	//==========================================================
	//GVD structurs.
	//==========================================================
	typedef struct 
	{
		BYTE		Version_Revision;
		BYTE		Version_Number;
		BYTE		Version_Minor;
		BYTE		Version_Major;
		BYTE		Version_spare[4];
	}ChangeSetVersion;

	typedef struct 
	{
		BYTE		Version_Minor;
		BYTE		Version_Major;
		BYTE		Version_spare[2];
	}BYTE_MajorMinor_Version;

	typedef struct 
	{
		UINT16		Version_Minor;
		UINT16		Version_Major;
	}UINT16_MajorMinor_Version_Reversed;

	typedef struct 
	{
		UINT16		Version_Major;
		UINT16		Version_Minor;
	}UINT16_MajorMinor_Version;

	typedef struct 
	{
		BYTE		Serial[6];
		BYTE		Spare[2];
	}BarCodeSerial;

	typedef struct 
	{
		BYTE		Serial[8];
		char		Spare[4];
	}PmbSerial;

	typedef struct 
	{
		BYTE		Serial[8];
		BYTE		Spare[4];
	}UlongNumber;


	typedef struct 
	{
		//Internal Struct
		ChangeSetVersion FWVersion;    //8 bytes
		ChangeSetVersion CoreVersion;  //8 bytes
		ChangeSetVersion USBVersion;   //8 bytes
		ChangeSetVersion DriversVersion;  //8 bytes
		ChangeSetVersion LiguriaVersion;  //8 bytes
		BYTE_MajorMinor_Version MlBinVersion; //4 bytes
		BYTE_MajorMinor_Version ClibRoVersion; //4 bytes
		BYTE_MajorMinor_Version ProjPatternVersion; //4 bytes
		BYTE_MajorMinor_Version ProjGainVersion; //4 bytes
		BYTE		FrozaVersion; //1 byte
		BYTE_MajorMinor_Version CalibVersion; //4 bytes
		UINT		AsicMask; //4 bytes
		UINT		EepromPmbVersion; //4 bytes
		BYTE_MajorMinor_Version Eeprom2CamVersion; //4 bytes
		BYTE_MajorMinor_Version Eeprom2CamAssemblyVersion; //4 bytes
		UINT		TesterMode; //4 bytes
		UINT16_MajorMinor_Version_Reversed  StarpState; //4 bytes
		UINT FPGAVersion; //4 bytes
		UINT RGBSensorVersion; //4 bytes
		BarCodeSerial ModuleSerialVersion; //8 bytes
		PmbSerial PmbSerialVersion; //12 bytes
		BYTE Serial2Cam[16];		//16 bytes   //if Eeprom2CamVersion is >= from 4.3 take 14 else 13. 
		UINT16_MajorMinor_Version FlashRoVersion; //4 bytes
		UINT16_MajorMinor_Version FlashRwVersion; //4 bytes
		BYTE spare[4]; //4 bytes
		UINT LiguriaApiVersion; //4 bytes
		UINT	AsicVersion; //4 bytes
		THWType	HwType; //4 bytes
		UINT16_MajorMinor_Version_Reversed OemId; //4 bytes
		BYTE		LDVersion; //1 byte
		BYTE		LDRevision;  //1 byte
		BYTE MEMSVersion[2]; //2 bytes
		BYTE DFUVersion[2]; //2 bytes
		char AsicId[32]; //32 bytes
	}TGVDInfo;

	 

	struct ConnectivityI2CResult
    {
        // Is I2c Is Valid
		bool IsValid;

        // True for RGB check Failed
        bool IsRgbFailed;

        // True for Depth check Failed
		bool IsDepthFailed;

        // True for Liguria check Failed
		bool IsLiguriaFailed;

        // True for Laser check Failed
		bool IsLaserFailed;

        // True for EepromPmb check Failed
		bool IsEepromPmbFailed;

        // True for pmb2Cam check Failed
        bool IsEeprom2CamFailed;

        // True for Eeprom Data Check Failed
        bool IseepromDataFailed;

		ConnectivityI2CResult():IsValid(false),
							   IsRgbFailed(false),
							   IsDepthFailed(false),
							   IsLiguriaFailed(false),
							   IsLaserFailed(false),
							   IsEepromPmbFailed(false),
							   IsEeprom2CamFailed(false),
							   IseepromDataFailed(false){};
	};



	struct ConnectivityRgbResult
    {
        // Is Rgb Valid
		bool IsValid;
      
        // Is TsvrEphy register valid
		bool IsTSvrEphyFailed;
       
        // Is TsvrLastLongPacket Register Valid
        bool IsTSvrLastLongPacketFailed ;
       
        // Is TsvrGoodFrame Register Valid
        bool IsTSvrGoodFrameFailed;
      
        // Return Light Value
       UINT ReturnStatValue;


	   //ctor
	   ConnectivityRgbResult():IsValid(false),
							   IsTSvrEphyFailed(false),
							   IsTSvrLastLongPacketFailed(false),
							   IsTSvrGoodFrameFailed(false),
							   ReturnStatValue(0){};
	};


	struct ConnectivityDepthResult
    {
        //Is Depth Valid
		bool IsValid;
        //Is PSvrEphy Register Valid
		bool IsPSvrEphyFailed;
        //Is PSvrLastLongPacket Register Valid
        bool IsPSvrLastLongPacketFailed;
        //Is PSvrGoodFrame Register Valid
		bool IsPSvrGoodFrameFailed;
        // Return calculated IR stat Value
		UINT ReturnStatValue;

		//ctor
	    ConnectivityDepthResult():IsValid(false),
							   IsPSvrEphyFailed(false),
							   IsPSvrLastLongPacketFailed(false),
							   IsPSvrGoodFrameFailed(false),
							   ReturnStatValue(0){};
	 };


	struct ConnectivityCheckInfo
    {
		IN UINT DPTLightValue;
		IN UINT RGBLightValue;
		OUT ConnectivityI2CResult connectivityI2CResult;
		OUT ConnectivityRgbResult connectivityRgbResult;
		OUT ConnectivityDepthResult connectivityDepthResult;
	};

//OBSOLETE:

	/*OBSOLETE*/typedef struct VersionInfo 
		/*OBSOLETE*/{
			/*OBSOLETE*/    char Reserved1[80];
			/*OBSOLETE*/    ivUINT32 version;
			/*OBSOLETE*/    char Reserved2[16];
			/*OBSOLETE*/    ivUINT32 serialHigh[4];
			/*OBSOLETE*/    char seriallow;
			/*OBSOLETE*/    char Reserved3[24];
			/*OBSOLETE*/}TVersionInfo;

}

#endif // _IVCAM_TYPES_H

