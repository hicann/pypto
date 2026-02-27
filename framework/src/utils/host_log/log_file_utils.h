/**
* Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file log_file_utils.h
 * \brief
 */

#include <climits>
#include <string>
#include <sys/stat.h>
#include <queue>
#include <dirent.h>

namespace npu::tile_fwk {
inline std::string GetRealPath(const std::string &path) {
    if (path.empty()) {
        return "";
    }
    if (path.size() >= PATH_MAX) {
        return "";
    }

    // PATH_MAX is the system marco, indicate the maximum length for file path
    // pclint check one param in stack can not exceed 1K bytes
    char resovedPath[PATH_MAX] = {0x00};

    std::string res;

    // path not exists or not allowed to read return nullptr
    // path exists and readable, return the resoved path
    if (realpath(path.c_str(), resovedPath) != nullptr) {
        res = resovedPath;
    }
    return res;
}

bool CreateSingleLevelDir(const char *dirPath) {
    std::string realPath = GetRealPath(dirPath);
    if (realPath.empty()) {
        int32_t ret = 0;
        ret = mkdir(dirPath, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);  // 755
        if (ret != 0 && errno != EEXIST) {
            return false;
        }
    }
    return true;
}

bool CreateMultiLevelDirectory(const std::string &directoryPath) {
    auto dirPathLen = directoryPath.length();
    if (dirPathLen >= PATH_MAX) {
        return false;
    }
    char tmpDirPath[PATH_MAX] = {};
    for (size_t i = 0; i < dirPathLen; ++i) {
        tmpDirPath[i] = directoryPath[i];
        if ((tmpDirPath[i] == '\\') || (tmpDirPath[i] == '/')) {
            if (access(tmpDirPath, F_OK) == 0) {
                continue;
            }
            if (!CreateSingleLevelDir(tmpDirPath)) {
                return false;
            }
        }
    }

    return CreateSingleLevelDir(directoryPath.c_str());
}

void RemoveFile(const std::string &path) {
    if (path.empty()) {
        return;
    }
    struct stat statBuf;
    if (lstat(path.c_str(), &statBuf) != 0) {
        return;
    }
    if (S_ISREG(statBuf.st_mode) == 0) {
        return;
    }
    (void)remove(path.c_str());
}

void LoadFileFromDir(const std::string &dirPath, const std::string &filePrefix, const std::string &fileSuffix,
                     std::queue<std::string> &files) {
    DIR *dir = opendir(dirPath.c_str());
    if (dir == nullptr) {
        return;
    }
    struct dirent *dirp = nullptr;
    while ((dirp = readdir(dir)) != nullptr) {
        if (dirp->d_name[0] == '.') {
            continue;
        }
        std::string fileName = dirp->d_name;
        if (fileName.find(filePrefix) != 0) {
            continue;
        }
        if (fileName.find(fileSuffix) == std::string::npos) {
            continue;
        }
        files.push(dirPath + "/" + fileName);
    }
    closedir(dir);
}
}
