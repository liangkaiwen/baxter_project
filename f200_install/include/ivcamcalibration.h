/*
 * ivcamcalibration.h
 *
 *  Created on: Oct 21, 2014
 *      Author: albert
 */

#ifndef IVCAMCALIBRATION_H_
#define IVCAMCALIBRATION_H_

#define IN
#define OUT

#include <libusb-1.0/libusb.h>
#include <pthread.h>
#include "InternalDefinitions.h"
//#include "IvcamTypes.h"
#include "ivcamParameterReader.h"

namespace ivcam_env
{

using namespace Ivcam;
//#define HW_MONITOR_BUFFER_SIZE  1000
//#define HW_MONITOR_COMMAND_SIZE 1000

#define WAIT_FOR_MUTEX_TIME_OUT 5000
#define DEF_TIME_TO_SLEEP_MS 	2000
//#define NUM_OF_CALIBRATION_COEFFS 64

#define ms2ns(X) ((X)*1000000)
#define ms2s(X)  ((X)/1000)



/*
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


typedef struct IRThermalLoopParams
{
	IRThermalLoopParams(){
			IRThermalLoopEnable = 1;
			TimeOutA			 = 10000;
			TimeOutB			 = 0;
			TimeOutC			 = 0;
			TransitionTemp	 = 3;
			TempThreshold		 = 2;
			HFOVsensitivity	 = 0.025f;
			FcxSlopeA			 = -0.003696988f;
			FcxSlopeB			 = 0.005809239f;
			FcxSlopeC			 = 0;
			FcxOffset			 = 0;
			UxSlopeA			 = -0.000210918f;
			UxSlopeB			 = 0.000034253955f;
			UxSlopeC			 = 0;
			UxOffset			 = 0;
			LiguriaTempWeight	 = 1;
			IrTempWeight		 = 0;
			AmbientTempWeight	 = 0;
			Param1			 = 0;
			Param2			 = 0;
			Param3			 = 0;
			Param4			 = 0;
			Param5			 = 0;
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
*/
typedef struct AsicCoefficiants
{
	float CoefValueArray[NUM_OF_CALIBRATION_COEFFS];
}TAsicCoefficiants;

class IVCAMController : public IVCAMParameterReader
{
public:
	IVCAM_RESULT OpenIVCAMMonitor();
	void CloseIVCAMMonitor();

	CalibrationParameters *getParameters();
	void SetDepthResolution(int w, int h);
	bool StartTempCompensationLoop(void);
	void StopTempCompensationLoop(void);
	void RollUSBDevices();
	~IVCAMController();


private:
	void printdev(libusb_device *dev);
	IVCAM_RESULT RetrieveRawIVCAMCalibrationData(IN OUT uint8_t *retBuffer, IN OUT size_t& len);
	void ProjectionCalibrate(void * param1, TCalibrationParameters *calprms);

	IVCAM_RESULT StartThread(IVCAMController *pThis);
	IVCAM_RESULT StopThread(void);
	static void *LaunchThread(void *pThis);
	IVCAM_RESULT TemperatureThread(void *threadArg);
	IVCAM_RESULT ReadTemperatures( TTemperatureData& temperatureData);
	IVCAM_RESULT GetMEMStemp(float& MEMStemp);
	IVCAM_RESULT GetIRtemp(int& IRtemp);
	IVCAM_RESULT GetCalibrationAsRawData(BYTE* data, int& bytesReturned);
	IVCAM_RESULT UpdateAsicCoeficiants(TAsicCoefficiants * AsicCoefficiants);
	IVCAM_RESULT FillUsbBuffer(IN int opCodeNumber, IN int param1, IN int param2,
												IN int param3, IN int param4, IN char* data, IN int dataLength,
												OUT char* bufferToSend,  OUT int &length/*, DataType dataType*/);

	IVCAM_RESULT PerfomAndSendHWmonitorCommand(IN OUT TIVCAMCommandParameters& CommandParameters);
	IVCAM_RESULT SendHWmonitorCommand(IN OUT TIVCAMCommandDetails& IVCAMCommandDetails);

	int executeCommand(uint8_t *out, size_t outSize, uint32_t &op, uint8_t *in, size_t &inSize);
	int prepareCommand(uint8_t *request, size_t &requestSize, uint32_t op,
						uint32_t param1=0, uint32_t param2=0, uint32_t param3=0, uint32_t param4=0,
						uint8_t *data=0, size_t dataLength=0);
	void dumpCommand(uint8_t *buf, size_t size);
	static int getVersionOfCalibration(uint8_t *validation, uint8_t *version);
	static void fillCalibrationData(OUT TCalibrationDataWithVersion &CalibrationData, IN int size, IN byte* data);
	static int bcdtoint(uint8_t* buf, int bufsize);

	CalibrationParameters *mCalibrationParameters;

	libusb_device_handle* mDev;
	pthread_mutex_t musbMutex;
	pthread_mutex_t mcalParamsMutex;
	bool mTempThreadRunning;
	pthread_t mTempThread;
	bool mUpdateCalibrationParameters;
	int mStopEventFd;
public:
	static int pthread_mutex_reltime_lock(pthread_mutex_t *m, int time);
};

}

#endif /* IVCAMCALIBRATION_H_ */
