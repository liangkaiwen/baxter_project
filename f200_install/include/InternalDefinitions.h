/*
 * InternalDefinitions.h
 *
 *  Created on: Nov 11, 2014
 *      Author: albert
 */

#ifndef INTERNALDEFINITIONS_H_
#define INTERNALDEFINITIONS_H_

#pragma once


#include "IvcamTypes.h"
#include "string.h"

using namespace std;
using namespace Ivcam;

#define MAX_SIZE_OF_CALIB_PARAM_BYTES   (800)
#define SIZE_OF_CALIB_PARAM_BYTES   (512)
#define SIZE_OF_CALIB_HEADER_BYTES   (4)

#define NUM_OF_CALIBRATION_COEFFS   (64)
#define MAX_NUM_OF_CONNECTED_CAMERAS (100)
#define CREATE_MUTEX_RETRY_NUM  (5)


#define LOG_INFO(msg)\
{\
	LogService * TheLog = LogService::GetInstance();						\
	if (TheLog)																\
		TheLog->Log(LogLevel_Info,msg);										\
}

#define LockRM(cameraIndex) Manager::GetInstance()->GetCameraRecursiveMutex(cameraIndex)->lock();
#define FreeRM(cameraIndex) Manager::GetInstance()->GetCameraRecursiveMutex(cameraIndex)->unlock();

#define FREE_RM_AND_RETURN_IF_FAIL(res,m_IndexOfCamera)	if(res != ivcamSuccess)															\
													{																					\
														Manager::GetInstance()->GetCameraRecursiveMutex(m_IndexOfCamera)->unlock();		\
														RETURN_ON_IVCAM_FAILURE(res);													\
													}


#define FREE_RM_AND_RETURN_IF_DEVICE_NOT_CONNECTED(res,IndexOfCamera)	if((res == ivcamColorDeviceNotConnected) || (res == ivcamDepthDeviceNotConnected))\
													{						\
														FreeRM(IndexOfCamera);				\
														RETURN_ON_IVCAM_FAILURE(res);			\
													}

#define RETURN_IF_FAILURE(res)	if (((TIVCAM_STATUS)res) != ivcamSuccess ) {	return res;	}

#ifdef DEBUG
#define DEBUG_PRINT(MsgToPrint)									OutputDebugStringA( MsgToPrint );

#define DEBUG_PRINT_ONE_PARAM(MsgToPrint, MsgParam)				{					\
	char MyMessage[256];															\
	sprintf_s(MyMessage , 256, MsgToPrint, MsgParam);								\
	OutputDebugStringA( MyMessage );												\
	cout<<MyMessage;																\
}

#define DEBUG_PRINT_TWO_PARAM(MsgToPrint, MsgP1,MsgP2)			{					\
	char MyMessage[256];															\
	sprintf_s(MyMessage , 256, MsgToPrint, MsgP1, MsgP2);							\
	OutputDebugStringA( MyMessage );												\
	cout<<MyMessage;																\
}


#define DEBUG_PRINT_THREE_PARAM(MsgToPrint, MsgP1,MsgP2,MsgP3)	{					\
	char MyMessage[256];															\
	sprintf_s(MyMessage , 256, MsgToPrint, MsgP1, MsgP2, MsgP3);					\
	OutputDebugStringA( MyMessage );												\
	cout<<MyMessage;																\
}
#else

#define DEBUG_PRINT(MsgToPrint)

#define DEBUG_PRINT_ONE_PARAM(MsgToPrint, MsgParam)

#define DEBUG_PRINT_TWO_PARAM(MsgToPrint, MsgP1,MsgP2)

#define DEBUG_PRINT_THREE_PARAM(MsgToPrint, MsgP1,MsgP2,MsgP3)

#endif



#define  _15FPSFrameSizeIn100Nan    666666;
#define  _30FPSFrameSizeIn100Nan    333333;
#define  _50FPSFrameSizeIn100Nan    200000;
#define  _60FPSFrameSizeIn100Nan    166666;
#define  _100FPSFrameSizeIn100Nan   100000;
#define  _200FPSFrameSizeIn100Nan   50000;
#define  _120FPSFrameSizeIn100Nan   83333;
#define  _400FPSFrameSizeIn100Nan   25000;
#define  _600FPSFrameSizeIn100Nan   16666;
#define  _1200FPSFrameSizeIn100Nan  8333;

#define UNIQUE_NUMBER 0xcafecafe;



#define IVCAM_COLOR_DEVICE_NAME	L"Cam RGB"      // RGB device name as it appear on device manager
#define IVCAM_DEPTH_DEVICE_NAME	L"Cam DPT"      // Depth device name as it appear on device manager
#define IVCAM_HW_MONITOR_DEVICE_NAME	L"MSFT100"      // Depth device name as it appear on device manager

#define IVCAM_COLOR_DEVICE_NEW_NAME_1	L"Intel(R) RealSense(TM) 3D camera RGB"	// RGB device new name as it appear on device manager
#define IVCAM_DEPTH_DEVICE_NEW_NAME_1	L"Intel(R) RealSense(TM) 3D camera"			// RGB device new name as it appear on device manager


#define IVCAM_COLOR_DEVICE_NEW_NAME	L"Intel(R) RealSense(TM) 3D Camera (Front F200) RGB"  // RGB device new name as it appear on device manager
#define IVCAM_DEPTH_DEVICE_NEW_NAME	L"Intel(R) RealSense(TM) 3D Camera (Front F200) Depth"  // RGB device new name as it appear on device manager
#define IVCAM_DEVICE_NEW_NAME	L"Intel(R) RealSense(TM) 3D Camera (Front F200)"
#define IVCAM_HW_MONITOR_DEVICE_NEW_NAME	L"Intel(R) RealSense(TM) 3D Camera (Front F200)"


#define IVCAM_VID_	"8086"
#define IVCAM_PID_	"0A66"
#define IVCAM_PID_LOWER_CASE_ "0a66"
#define IVCAM_DEVICE_NAME		L"IVCAM"


#define GVD_STRING_MAX_LENGTH 1024
#define GVD_INFO_FIELD_MAX_LENGTH 100

	typedef enum
	{
		HW_ColorNone ,
		HW_DepthNone ,
		HW_YUY,
		HW_RGB24,
		HW_RGB32,
        HW_BGR24,
        HW_BGR32,
		HW_INV_RAW16,
		HW_Z,
		HW_ZI,
		HW_R,
		HW_RI,
		HW_IntensityOnly,
		HW_RelativeIR,
		HW_IntensityOnly8BPP,
		HW_CALIBRATION_40BIT,
		HW_RAW_UINT16,
		HW_IRPACKEDRAW_12BPPDATA_UINT16,

		HW_IV08	,
		HW_IV16	,
		HW_IV24	,
		HW_IV40 ,


	} TCameraOutputType;


	class StreamsHandlerFSM;
	class Projection;
	class IVCAMHal;



	typedef enum
	{
		IVCAM_PROPERTY_MF_FIRMWARE_UPDATE_MODE			=  IVCAM_PROPERTY_CUSTOMIZED +1,
		IVCAM_PROPERTY_MF_TIME_STAMP_OPEN				=  IVCAM_PROPERTY_CUSTOMIZED +2
	}InternalProperties;



	typedef struct HWProfile
	{
		TCameraOutputType camOutType;
		TSurfaceResolution res;
		int fps;
	}THWProfile;


	typedef char TModuleSerial[6];
	typedef struct
	{
		uint8_t sw_ver1;
		uint8_t sw_ver2;
		uint8_t sw_ver_minor;
		uint8_t sw_ver_major;
		uint32_t sw_changeset;

	} TSwVer;

	typedef uint32_t TblVer;


	typedef struct
	{
		double Kc[3][3];
		double Distc[5];
		double Invdistc[5];
		double ImageSize[2];
		double Rmax;
		double Pp_full[3][4];
		double Kp[3][3];
		double Rp[3][3];
		double Tp[3];
		double Distp[5];
		double Invdistp[5];
		double Pt[3][4];
		double Distt[5];

	}TOldCalibrationData;


	typedef struct
	{
		float Kc[3][3];
		float Distc[5];
		float Invdistc[5];
		float ImageSize[2];
		float Rmax;
		float Pp_full[3][4];
		float Kp[3][3];
		float Rp[3][3];
		float Tp[3];
		float Distp[5];
		float Invdistp[5];
		float Pt[3][4];
		float Distt[5];

	}TOldSerielizedCalibrationData;

	typedef struct TesterData
	{
		int16_t TableValidation;
		int16_t TableVarsion;
		TTemperatureData TemperatureData;

		//OAC
		TOACOffsetData OACOffsetData;

		//IR Thermal Loop Params
		TIRThermalLoopParams ThermalLoopParams;

	}TTesterData;


	typedef struct AsicCoefficiants
	{
		float CoefValueArray[NUM_OF_CALIBRATION_COEFFS];
	}TAsicCoefficiants;






	enum HWmonitorCommands
	{
		HWmonitor_UpdateCalib = 0xBC,
		HWmonitor_GetIRTemp = 0x52,
		HWmonitor_GetMEMSTemp = 0x0A,
		HWmonitor_HWReset =  0x28,
		HWmonitor_GVD = 0x3B,
		HWmonitor_BIST = 0xFF,
		HWmonitor_GoToDFU = 0x80,
		HWmonitor_GetCalibrationTable = 0x3D,
		HWmonitor_DebugFormat = 0x0B,
		HWmonitor_TimeStempEnable = 0x0C,
		HWmonitor_GetPowerGearState = 0xFF,
		HWmonitor_SetDefaultControls = 0xA6,
		HWmonitor_GetDefaultControls = 0xA7,
		HWmonitor_GetFWLastError = 0x0E,
		HWmonitor_CheckI2cConnect = 0x4A,
		HWmonitor_CheckRGBConnect = 0x4B,
		HWmonitor_CheckDPTConnect = 0x4C,
		HWmonitor_USBEnumerationStatus = 0x12,
		HWmonitor_GetCurrColorIntegrationTime = 0x53
	};

	enum ACPICommands
	{
		ACPI_PowerOff,
		ACPI_PowerOn
	};


	typedef enum
	{
		CamType_Color,
		CamType_Depth,
		CamType_Both,
		CamType_NoneOrError
	} TCamType;

	typedef enum
	{
		ExpectedCamType_Color = 1,
		ExpectedCamType_Depth = 2,
		ExpectedCamType_Both  = 3,
		ExpectedCamType_None  = 0
	} TWinUsbExpectedCamType;

	typedef enum
	{
		ExistCamType_Color = 1,
		ExistCamType_Depth = 2,
		ExistCamType_Both  = 3,
		ExistCamType_None  = 0
	} TWinUsbExistCamType;

	typedef enum
	{
		MissingCamType_Color = 1,
		MissingCamType_Depth = 2,
		MissingCamType_Both  = 3,
		MissingCamType_None  = 0
	} TWinUsbMissingCamType;

	typedef enum
	{
		ectVGA = 0,
		ectQVGA = 1,
		ectHVGA = 2,
		ectVGA_640X360 = 3,
		ectRVGA = 4,
		ectRQVGA = 5,
		ectRHVGA = 6,
		ectRVGA_640X360 = 7,
		ectCalibTableEnd
	}ETCalibTable;


	struct TIVCAMCommandParameters
	{
		IN HWmonitorCommands CommandOp;
		IN  int Param1;
		IN  int Param2;
		IN  int Param3;
		IN  int Param4;
		IN  char data[HW_MONITOR_BUFFER_SIZE];
		IN  int sizeOfSendCommandData;
		IN  long TimeOut;
		IN  bool oneDirection;
		OUT  char recivedCommandData[HW_MONITOR_BUFFER_SIZE];
		OUT  int sizeOfRecivedCommandData;
		OUT  char recievedOPcode[4];
	};

	typedef enum Base{
		Hex,
		Dec,
	};




	//==========================================================
	//vendors controls definitions
	//==========================================================

	typedef struct
	{
		uint32_t UVC_AE_MODE_CONTROL;
		uint32_t UVC_AE_PRIORITY_CONTROL;
		uint32_t COLOR_EXPOSURE; //copy from fw if outo
		uint32_t COLOR_BACK_LIGHT_COMPENSATION;
		uint32_t COLOR_BRIGHTNESS	;
		uint32_t COLOR_CONTRAST;
		uint32_t COLOR_GAIN;
		uint32_t COLOR_POWER_LINE_FREQUENCY;
		uint32_t COLOR_HUE;
		uint32_t COLOR_SATURATION;
		uint32_t COLOR_SHARPNESS;
		uint32_t COLOR_GAMMA;
		uint32_t COLOR_WHITE_BALANCE;  // copy from FW if outo
		uint32_t UVC_WHITE_BALANCE_TEMPERATURE_AUTO_CONTROL;    // 1 - outo 0 - manual
		uint32_t spare[8];  // copy from FW
	}
	UVCAptinaControlSet;

	typedef struct
	{
		uint32_t EU_CONTROL_UNDEFINED;	 //copy from fw
		uint32_t DEPTH_LASER_POWER;
		uint32_t DEPTH_ACCURACY;
		uint32_t DEPTH_MOTION_VS_RANGE_TRADE;
		uint32_t DEPTH_POWER_GEAR;
		uint32_t DEPTH_FILTER_OPTION;
		uint32_t DEPTH_CONFIDENCE_THRESHOLD;
		uint32_t DEPTH_FIRMWARE_UPDATE_MODE;
		uint32_t DEPTH_CALIBRATION_DATA;
		uint32_t DEPTH_TIMESTAMP_ENABLE;
		uint32_t DEPTH_LOW_CONFIDENCE_VALUE;
		uint32_t EU_SPARE;    //copy from fw
		uint32_t DEPTH_VERSION;
		uint32_t spare[8];//copy from fw
	}
	UVCExtControlSet;



	typedef struct{
		uint8_t		m_IsValid;											// if 0 - this default string is allocated
		uint8_t		m_Length;											// The maximum length is 60.
		uint16_t   	m_Padding;  										// Aligning the structure.
		uint8_t		m_String[MAX_VENDOR_SPECIFIC_STRING_SIZE];		// Container for user string. Must be filled according to the USB protocol.
	}TCamControlString;


	typedef	struct {
		uint32_t m_OemId;											// OEM Id
		uint16_t m_VendorId;										// = 0x8086,
		uint16_t m_DeviceId;										// = 0x0A66,
		TCamControlString	m_ControlStringDescriptorDPT;			// = "DPT",
		TCamControlString	m_ControlStringDescriptorRGB;			// = "RGB",
		TCamControlString	m_ControlStringDescriptorDevName;		// = "IVCAM",
		TCamControlString	m_ControlStringDescriptorHWM;			// = "HW_MONITOR",
		uint8_t spare[256];
	}SpecificVendorParams;


	typedef struct{
		SpecificVendorParams vendorSpecific;
		UVCExtControlSet euCtrlSet;
		UVCAptinaControlSet uvcAptinaCtrlSet;
	}TCtrlContainerTable;

struct FW_REV
{
	union
	{
		unsigned short value;
		struct
		{
			unsigned short Patch :5; //patch 5 bits 1-32
			unsigned short Minor :7; //minor 7 bits 1-128
			unsigned short Major :4; //Major 4 bits 1-16
		} Revision;
	} u;
};


typedef enum CreateOpenMutexStatus{
	Mutex_Succeed,
	Mutex_TotalFailure,
	Mutex_AlreadyExist
};


enum IvcamMessagesType
{
	ColorFrameFailureMsg			= 0,
	DepthFrameFailureMsg			= 1,
	EXIT_MSG						= 0xFF
};

typedef struct
{
	EventCallback eventCallback;
	void* userContext;
}EventCallbackData;

#endif /* INTERNALDEFINITIONS_H_ */
