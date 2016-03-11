#pragma once

// template <class T>
// class Calibration;
#include "Calibration.h"
using namespace Ivcam;

#ifndef WIN32
#define IVCAM_DLL_API
#endif


class Projection
{
public:

	IVCAM_DLL_API Projection(int indexOfCamera, bool RunThermalLoop = false);
	//Projection(const Projection &old); // disallow copy constructor
	//const Projection &operator=(const Projection &old); //disallow assignment operator
	IVCAM_DLL_API ~Projection(void);

	// ----------------------Projection API----------------------------------------//
	IVCAM_DLL_API static Projection*  GetInstance();

	/**The method return the size of the projection */
	//Input: none
	//Outeput: size - the size of the projection
	//Return: TIVCAM_STATUS
	TIVCAM_STATUS IVCAM_DLL_API GetProjectionSize(OUT int &size);


	/**The method serialize of the projection object into buffer of bytes */
	//Output: data - the buffer of serialize projection, should be allocated with the projection size
	//        size - the size of the projection
	//Return: TIVCAM_STATUS
	TIVCAM_STATUS IVCAM_DLL_API GetSerializedProjectionData(OUT int &size,OUT BYTE *data);


	/**The method deserialize of the projection object */
	//Input: data - the serialize projection.
	//Return: TIVCAM_STATUS
	TIVCAM_STATUS IVCAM_DLL_API SetSerializedProjectionData(IN BYTE *data);


	/**The method save the projection object to file as serialize data */
	//Input: fileName - full path of the file to save the projection.
	//Return: TIVCAM_STATUS
	TIVCAM_STATUS IVCAM_DLL_API SaveProjectionToFile(IN const string& fileName);


	/**The method construct the projection object with serialize data that saved to file*/
	//Input: fileName - full path of the file.
	//Return: TIVCAM_STATUS
	TIVCAM_STATUS IVCAM_DLL_API SetProjectionFromFile(IN const string& fileName);


	/**The method return the UV map of the depth image*/
	//Input: npoints           - number of points to be maped (can be used to map part of the image).
	//		 pos2d	           - array of 3D points in the the size of npoints   
	//						     pos2d.z is the depth in MM units and pos2d.x and pos2d.y are the index of the pixels.
	//		 isUVunitsRelative - if true  - output UV units are in the range of 0-1 and relative to the image size  
	//							 if false - output UV units are absolute.
	//Output: posc   - array of 3D points in the size of npoint pos2d.x and pos2d.y are the UV map of pixel (x,y)
	//return: TIVCAM_STATUS 
	TIVCAM_STATUS IVCAM_DLL_API MapDepthToColorCoordinates(IN unsigned int npoints,IN Point3DF32 *pos2d, OUT Point2DF32 *posc, IN bool isUVunitsRelative = true, TCoordinatSystemDirection coordinatSystemDirection = LeftHandedCoordinateSystem);


	/**The method return the UV map of the depth image*/
	//Input: width             - width of the image.
	//		 height			   - height of the image.
	//		 pSrcDepth	       - array of input depth in MM in the size of width*height.   
	//		 isUVunitsRelative - if true  - output UV units are in the range of 0-1 and relative to the image size  
	//							 if false - output UV units are absolute.
	//Output: pDestUV          - array of output UV map should be allocated in the size of width*height*2.
	//return: TIVCAM_STATUS 
	TIVCAM_STATUS IVCAM_DLL_API MapDepthToColorCoordinates(IN unsigned int width, IN unsigned int height, IN INT16* pSrcDepth, OUT float* pDestUV, IN bool isUVunitsRelative = true, TCoordinatSystemDirection coordinatSystemDirection = LeftHandedCoordinateSystem);


	/**The method convert the depth to coordinates in real world*/
	//Input: npoints           - number of points to be converted (can be used to convert part of the image).
	//		 pos2d	           - array of 3D points in the the size of npoints   
	//						     pos2d.z is the depth in MM units and pos2d.x and pos2d.y are the index of the pixels.
	//Output: pos3d            - array of 3D points in the size of npoint pos2d.x pos2d.y pos2d.z are the coordinates in real world of pixel (x,y)
	//Return: TIVCAM_STATUS 
	TIVCAM_STATUS IVCAM_DLL_API ProjectImageToRealWorld(IN unsigned int npoints, IN Point3DF32 *pos2d, OUT Point3DF32 *pos3d, TCoordinatSystemDirection coordinatSystemDirection = LeftHandedCoordinateSystem);


	/**The method convert the depth to coordinates in real world*/
	//Input: width             - width of the image.
	//		 height			   - height of the image.
	//		 pSrcDepth	       - array of input depth in MM in the size of width*height.   
	//Output: pDestXYZ         - array of output XYZ coordinates in real world of the corresponding to the input depth image should be allocated in the size of width*height*3.
	//return: TIVCAM_STATUS 
	TIVCAM_STATUS IVCAM_DLL_API ProjectImageToRealWorld(IN unsigned int width, IN unsigned int height, IN UINT16* pSrcDepth, OUT float* pDestXYZ, TCoordinatSystemDirection coordinatSystemDirection = LeftHandedCoordinateSystem);


	/**The method convert the depth in 1/32 MM units to MM*/
	//Input: d                 - depth in 1/32 MM units.
	//return: float            - depth in MM units.
	float IVCAM_DLL_API ConvertDepth_Uint16ToMM( IN UINT16 d);

	/**The method reset the projection object*/
	void IVCAM_DLL_API Free();


	// ----------------------For internal use----------------------------------------//
	
	bool Init();
	TIVCAM_STATUS Depth2Vertices(IN void* RawPointer, OUT TIVCAMStreamWrapper* streamWrapper);
	TIVCAM_STATUS depthToColor(IN void* RawPointer, OUT TIVCAMStreamWrapper* streamWrapper);
	
	bool IsInitialized() {return m_isInitialized;}
	void SetDepthResolution(int width, int height) { m_currentDepthWidth=width; m_currentDepthHeight=height;}
	void SetColorResolution(int width, int height) { m_currentColorWidth=width; m_currentColorHeight=height;}
	int GetColorWidth(){return m_currentColorWidth;}
	int GetColorHeight(){return m_currentColorHeight;}
	int GetDepthWidth(){return m_currentDepthWidth;}
	int GetDepthHeight(){return m_currentDepthHeight;}

	//float convertDepth_Uint16ToMM(UINT16* pSrcDepthUint16,  float* pDestDepthMM);

	TIVCAM_STATUS QueryProperty(Property label, OUT float &value);

	//Start function of Thermal loop thread.Thread
	//Will poll temperature each X seconds and make required changes to Calibration table.
	//Also inform users that calib table has changed and they need to redraw it.
	void CallThermalLoopThread(); 
	int GetIndexOfCamera(){return m_IndexOfCamera;};
	Calibration<float>* GetCalibrationObject(){return &m_calibration;};

	TIVCAM_STATUS InitializeThermalData(IN TTemperatureData TemperatureData, IN TIRThermalLoopParams ThermalLoopParams);

	TIVCAM_STATUS GetThermalData(OUT TTemperatureData &TemperatureData, OUT TIRThermalLoopParams &ThermalLoopParams);

	typedef struct 
	{
		UINT32 depthWidth;
		UINT32 depthHeight;
        UINT32 colorWidth;
		UINT32 colorHeight;
		UINT32 nParams;
		TCalibrationDataWithVersion calibrationParams;
	}ProjectionParams;


	TIVCAM_STATUS ThermalLoopKilled();

private:


	int m_IndexOfCamera;	
	bool m_isCalibOld;					
	//static Projection* m_pInstance;
	//static int m_nRefCount;


private:
	Calibration<float>			m_calibration;
    bool						m_isInitialized;
	int							m_currentDepthWidth;
	int							m_currentDepthHeight;
    int							m_currentColorWidth;
	int							m_currentColorHeight;
	TCalibrationDataWithVersion	m_calibrationData;

	bool						m_RunThermalLoop;
	bool						m_IsThermalLoopOpen;
	//Projection(const Projection&);
};

