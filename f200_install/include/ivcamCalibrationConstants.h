/*
 * ivcamCalibrationConstants.h
 *
 *  Created on: Oct 21, 2014
 *      Author: albert
 */

#ifndef IVCAMCALIBRATIONCONSTANTS_H_
#define IVCAMCALIBRATIONCONSTANTS_H_

#define IVCAM_VID                       0x8086
#define IVCAM_PID                       0x0A66
#define IVCAM_MONITOR_INTERFACE         0x4
#define IVCAM_MONITOR_ENDPOINT_OUT      0x1
#define IVCAM_MONITOR_ENDPOINT_IN       0x81
#define IVCAM_MONITOR_MAGIC_NUMBER      0xcdab
#define IVCAM_MONITOR_HEADER_SIZE       (sizeof(uint32_t)*6)
#define IVCAM_MONITOR_TIMEOUT_MS		1000

#define IVCAM_MIN_SUPPORTED_VERSION 13

#define IVCAM_MONITOR_MAX_BUFFER_SIZE 1024

namespace ivcam_env {

enum IvcamMonitorCommand
    {
        UpdateCalib = 0xBC,
        GetIRTemp = 0x52,
        GetMEMSTemp = 0x0A,
        HWReset =  0x28,
        GVD = 0x3B,
        BIST = 0xFF,
        GoToDFU = 0x80,
        GetCalibrationTable = 0x3D,
        DebugFormat = 0x0B,
        TimeStempEnable = 0x0C,
        GetPowerGearState = 0xFF,
        SetDefaultControls = 0xA6,
        GetDefaultControls = 0xA7,
        GetFWLastError = 0x0E,
        CheckI2cConnect = 0x4A,
        CheckRGBConnect = 0x4B,
        CheckDPTConnect = 0x4C
    };

}

#endif /* IVCAMCALIBRATIONCONSTANTS_H_ */
