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
 * \file file_utils.cpp
 * \brief
 */

#include "interface/utils/file_utils.h"
#include <fstream>
#include <fcntl.h>
#include <climits>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <dlfcn.h>
#include <ftw.h>
#include "interface/utils/log.h"

namespace npu::tile_fwk {
namespace {
const int FILE_AUTHORITY = 0640;
}

bool FileExist(const std::string &filePath) {
    return !RealPath(filePath).empty();
}

std::string RealPath(const std::string &path) {
    if (path.empty()) {
        ALOG_INFO("path string is nullptr.");
        return "";
    }
    if (path.size() >= PATH_MAX) {
        ALOG_INFO("file path ", path.c_str(), " is too long.");
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
    } else {
        ALOG_INFO("path ", path.c_str(), " is not exist.");
    }
    return res;
}

bool GetFileSize(const std::string& filePath, uint32_t &fileSize) {
    if (RealPath(filePath).empty()) {
        return false;
    }
    std::ifstream file(filePath, std::ios::binary | std::ios::ate); // 打开文件，定位到文件末尾
    if (!file.is_open()) {
        return false;
    }
    fileSize = file.tellg();
    file.close();
    return true;
}

uint32_t GetFileSize(const std::string& filePath) {
    uint32_t fileSize = 0;
    (void) GetFileSize(filePath, fileSize);
    return fileSize;
}

bool CreateDir(const std::string &directoryPath) {
    if (!RealPath(directoryPath).empty()) {
        return true;
    }
    int32_t ret = mkdir(directoryPath.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);  // 755
    if (ret != 0 && errno != EEXIST) {
        ALOG_WARN("Creat dir[", directoryPath.c_str(), "] failed, reason is ", strerror(errno));
        return false;
    }
    return true;
}

bool JudgeEmptyAndCreateDir(char tmpDirPath[], const std::string &directoryPath) {
    std::string realPath = RealPath(tmpDirPath);
    if (realPath.empty()) {
        int32_t ret = 0;
        ret = mkdir(tmpDirPath, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);  // 755
        if (ret != 0 && errno != EEXIST) {
            ALOG_WARN("Creat dir[", directoryPath.c_str(), "] failed, reason is ", strerror(errno));
            return false;
        }
    }
    return true;
}

bool CreateMultiLevelDir(const std::string &directoryPath) {
    auto dirPathLen = directoryPath.length();
    if (dirPathLen >= PATH_MAX) {
        ALOG_WARN("Path[", directoryPath.c_str(), "] is too long, it must be less than ", PATH_MAX);
        return false;
    }
    char tmpDirPath[PATH_MAX] = {0};
    int32_t ret;
    for (size_t i = 0; i < dirPathLen; ++i) {
        tmpDirPath[i] = directoryPath[i];
        if ((tmpDirPath[i] == '\\') || (tmpDirPath[i] == '/')) {
            if (access(tmpDirPath, F_OK) == 0) {
                continue;
            }
            if (!JudgeEmptyAndCreateDir(tmpDirPath, directoryPath)) {
                return false;
            }
        }
    }

    std::string path = RealPath(directoryPath);
    if (path.empty()) {
        ret = mkdir(directoryPath.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);  // 755
        if (ret != 0 && errno != EEXIST) {
            ALOG_WARN("Creat dir[", directoryPath.c_str(), "] failed, reason is: ", strerror(errno));
            return false;
        }
    }

    ALOG_DEBUG("Create multi level dir [", directoryPath.c_str(), "] successfully.");
    return true;
}

void DeleteFile(const std::string &path)
{
    if (path.empty()) {
        ALOG_WARN("File name is empty.");
        return;
    }
    struct stat statBuf;
    if (lstat(path.c_str(), &statBuf) != 0) {
        ALOG_WARN("Stat file[", path.c_str(), "] failed.");
        return;
    }
    if (S_ISREG(statBuf.st_mode) == 0) {
        ALOG_WARN("[", path.c_str(), "] is not a file.");
        return;
    }
    int res = remove(path.c_str());
    if (res != 0) {
        ALOG_WARN("Delete file[", path.c_str(), "] failed.");
    }
}

bool ReadJsonFile(const std::string &file, nlohmann::json &jsonObj) {
    std::string path = RealPath(file);
    if (path.empty()) {
        ALOG_WARN("File path [", file.c_str(), "] does not exist");
        return false;
    }
    std::ifstream ifStream(path);
    try {
        if (!ifStream.is_open()) {
            ALOG_WARN("Open ", file.c_str(), " failed, file is already open");
            return false;
        }

        ifStream >> jsonObj;
        ifStream.close();
    } catch (const std::exception &e) {
        ALOG_WARN("Fail to convert file[", path.c_str(), "] to Json. Exception message is .", e.what());
        ifStream.close();
        return false;
    }

    return true;
}

bool ReadBytesFromFile(const std::string &filePath, std::vector<char> &buffer)
{
    std::string realPath = RealPath(filePath);
    if (realPath.empty()) {
        ALOG_WARN_F("Bin file path[%s] is not valid.", filePath.c_str());
        return false;
    }

    std::ifstream ifStream(realPath.c_str(), std::ios::binary | std::ios::ate);
    if (!ifStream.is_open()) {
        ALOG_WARN_F("read file %s failed.", filePath.c_str());
        return false;
    }
    try {
        std::streamsize size = ifStream.tellg();
        if (size <= 0 || size > INT_MAX) {
            ifStream.close();
            ALOG_WARN_F("File size %ld is not within the range: (0, %d].", size, INT_MAX);
            return false;
        }

        ifStream.seekg(0, std::ios::beg);

        buffer.resize(size);
        ifStream.read(&buffer[0], size);
        ALOG_DEBUG_F("Release file(%s) handle.", realPath.c_str());
        ifStream.close();
        ALOG_DEBUG_F("Read size: %ld.", size);
    } catch (const std::ifstream::failure& e) {
        ALOG_WARN_F("Fail to read file %s. Exception: %s.", filePath.c_str(), e.what());
        ifStream.close();
        return false;
    }
    return true;
}

bool IsPathExist(const std::string& path)
{
    if (path.empty()) {
        return false;
    }

    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0);
}

std::vector<std::string> GetFiles(const std::string& path, const std::string& ext) {
    std::vector<std::string> files;
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        ALOG_WARN("Open directory [", path.c_str(), "] failed");
        return files;
    }

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        std::string fileName = ent->d_name;
        if (fileName == "." || fileName == "..") {
            continue;
        }

        // 检查文件扩展名
        if (!ext.empty()) {
            size_t pos = fileName.rfind('.');
            if (pos == std::string::npos) {
                continue;  // 没有扩展名
            }
            std::string fileExt = fileName.substr(pos + 1);
            // 转换为小写进行比较
            std::transform(fileExt.begin(), fileExt.end(), fileExt.begin(), ::tolower);
            std::string targetExt = ext;
            std::transform(targetExt.begin(), targetExt.end(), targetExt.begin(), ::tolower);
            if (fileExt != targetExt) {
                continue;
            }
        }

        files.push_back(fileName);
    }
    closedir(dir);

    std::sort(files.begin(), files.end());
    return files;
}

void SaveFile(const std::string &filePath, const std::vector<uint8_t> &data) {
    FILE *file = fopen(filePath.c_str(), "wb");
    if (file == nullptr) {
        ALOG_WARN_F("Open file [%s] failed.", filePath.c_str());
        return;
    }
    fwrite(data.data(), 1, data.size(), file);
    fclose(file);
}

bool SaveFile(const std::string &filePath, const uint8_t *data, size_t size) {
    FILE *file = fopen(filePath.c_str(), "wb");
    if (file == nullptr) {
        ALOG_WARN_F("Open file [%s] failed.", filePath.c_str());
        return false;
    }
    fwrite(data, 1, size, file);
    fclose(file);
    return true;
}

void SaveFileSafe(const std::string &filePath, const uint8_t *data, size_t size) {
    auto tmpfile = filePath + ".tmp";
    if (SaveFile(tmpfile, data, size)) {
        Rename(tmpfile, filePath);
    }
}

void Rename(const std::string &oldPath, const std::string &newPath) {
    if (rename(oldPath.c_str(), newPath.c_str()) != 0) {
        ALOG_WARN_F("Rename file %s to %s failed.", oldPath.c_str(), newPath.c_str());
    }
}

bool DumpFile(const char *data, const size_t size, const std::string &filePath) {
    // dump bin file
    std::ofstream outFile(filePath, std::ios::binary);
    if (!outFile) {
        ALOG_ERROR_F("Failed open file %s.", filePath.c_str());
        return false;
    }
    outFile.write(data, size);
    outFile.close();
    ALOG_INFO_F("Bin file[%s] has been dumped.", filePath.c_str());
    return true;
}

bool DumpFile(const std::vector<uint8_t> &data, const std::string &filePath) {
    return DumpFile(reinterpret_cast<const char *>(data.data()), data.size(), filePath);
}

bool DumpFile(const std::string &text, const std::string &filePath) {
    return DumpFile(text.data(), text.size(), filePath);
}

std::vector<uint8_t> LoadFile(const std::string &filePath) {
    std::vector<uint8_t> binary;
    std::string realPath = RealPath(filePath);
    if (realPath.empty()) {
        ALOG_WARN_F("Bin file path[%s] is not valid.", filePath.c_str());
        return binary;
    }

    FILE *file = fopen(filePath.c_str(), "rb");
    if (file != nullptr) {
        fseek(file, 0, SEEK_END);
        int size = ftell(file);
        binary.resize(size);
        fseek(file, 0, SEEK_SET);
        fread(binary.data(), 1, size, file);
        fclose(file);
    }
    return binary;
}

static int RemoveFile(const char* path, const struct stat* sb, int flag, struct FTW* ftwbuf) {
    (void)sb;
    (void)ftwbuf;
    if (flag == FTW_F) {
        return remove(path);
    } else if (flag == FTW_DP) {
        return rmdir(path);
    }
    return 0;
}

bool DeleteDir(const std::string& directoryPath) {
    constexpr int limit = 64;
    int ret = nftw(directoryPath.c_str(), RemoveFile, limit, FTW_DEPTH | FTW_PHYS);
    if (ret != 0) {
        ALOG_WARN("Delete dir[", directoryPath.c_str(), "] failed, reason is ", ret);
        return false;
    }
    return true;
}

bool FcntlLockFile(const int fd, const int type) {
    struct flock lock;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    lock.l_type = type;

    // lock or unlock
    return fcntl(fd, F_SETLK, &lock) == 0;
}

FILE* LockAndOpenFile(const std::string &lockFilePath) {
    FILE *fp = fopen(lockFilePath.c_str(), "a+");
    if (fp == nullptr) {
        return nullptr;
    }
    (void)chmod(lockFilePath.c_str(), FILE_AUTHORITY);
    if (!FcntlLockFile(fileno(fp), F_WRLCK)) {
        ALOG_WARN("Fail to lock file:", lockFilePath.c_str());
        fclose(fp);
        return nullptr;
    }
    ALOG_INFO("Lock file successfully.", lockFilePath.c_str());
    return fp;
}

void UnlockAndCloseFile(FILE *fp) {
    if (fp == nullptr) {
        return;
    }
    (void)FcntlLockFile(fileno(fp), F_UNLCK);
    fclose(fp);
    fp = nullptr;
}

bool CopyFile(const std::string &srcPath, const std::string &dstPath) {
    std::ifstream src(srcPath, std::ios::binary);
    std::ofstream dst(dstPath, std::ios::binary);

    if (!src.is_open() || !dst.is_open()) {
        ALOG_WARN("Fail to open file:", srcPath.c_str(), ", ", dstPath.c_str());
        return false;
    }

    dst << src.rdbuf();
    src.close();
    dst.close();
    return true;
}

std::string GetCurrentSharedLibPath() {
    static std::string currentLibPath;
    if (!currentLibPath.empty()) {
        return currentLibPath;
    }

    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(GetCurrentSharedLibPath), &info)) {
        currentLibPath = std::string(info.dli_fname);
        int32_t pos = currentLibPath.rfind('/');
        if (pos >= 0) {
            currentLibPath = currentLibPath.substr(0, pos);
        }
    }
    return currentLibPath;
}

std::string GetCurRunningPath() {
    constexpr size_t size = 1024;
    char buffer[size] = {};
    std::string cwd = getcwd(buffer, size);
    if (cwd.empty()) {
        ALOG_ERROR_F("failed to call getcwd()");
        return "";
    }
    return cwd;
}

void RemoveOldestDirs(const std::string &path, const std::string &prefix, int left) {
    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        ALOG_WARN_F("failed to opendir: %s", path.c_str());
        return;
    }

    int32_t dirNum{0};
    struct dirent *entry;
    std::map<long long, std::string, std::less<>> timeList;
    while ((entry = readdir(dir)) != nullptr) {
        if (strncmp(entry->d_name, prefix.c_str(), prefix.size()) != 0) {
            continue;
        }

        std::string dirName(entry->d_name);
        std::string fullPath = path + "/" + dirName;
        struct stat statBuf;
        if (stat(fullPath.c_str(), &statBuf) == 0 && S_ISDIR(statBuf.st_mode)) {
            ++dirNum;
            long long tmpTime = static_cast<long long>(statBuf.st_mtime);
            timeList[tmpTime] = fullPath;
        }
    }
    closedir(dir);

    for (auto it = timeList.begin(); dirNum > left && it != timeList.end(); --dirNum, ++it) {
        DeleteDir(it->second);
    }
}
}  // namespace npu::tile_fwk
