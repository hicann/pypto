/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file file_utils.h
 * \brief
 */

#pragma once

#include <nlohmann/json.hpp>
#include <string>

namespace npu::tile_fwk {
bool FileExist(const std::string &filePath);
std::string RealPath(const std::string& path);
bool GetFileSize(const std::string &filePath, uint32_t &fileSize);
uint32_t GetFileSize(const std::string& filePath);
bool CreateDir(const std::string &directoryPath);
bool DeleteDir(const std::string& directoryPath);
bool CreateMultiLevelDir(const std::string &directoryPath);
void DeleteFile(const std::string &path);
bool ReadJsonFile(const std::string& file, nlohmann::json& jsonObj);
bool ReadBytesFromFile(const std::string &filePath, std::vector<char> &buffer);
bool IsPathExist(const std::string& path);
std::vector<std::string> GetFiles(const std::string& path, const std::string& ext);
void SaveFile(const std::string &filePath, const std::vector<uint8_t> &data);
bool SaveFile(const std::string &filePath, const uint8_t *data, size_t size) __attribute__ ((warn_unused_result));
void SaveFileSafe(const std::string &filePath, const uint8_t *data, size_t size);
bool DumpFile(const char *data, const size_t size, const std::string &filePath);
bool DumpFile(const std::vector<uint8_t> &data, const std::string &filePath);
bool DumpFile(const std::string &text, const std::string &filePath);
void Rename(const std::string &oldPath, const std::string &newPath);
std::vector<uint8_t> LoadFile(const std::string &filePath);
FILE* LockAndOpenFile(const std::string &lockFilePath);
void UnlockAndCloseFile(FILE *fp);
bool CopyFile(const std::string &srcPath, const std::string &dstPath);
std::string GetCurrentSharedLibPath();
std::string GetCurRunningPath();
void RemoveOldestDirs(const std::string &path, const std::string &prefix, int left);
}
