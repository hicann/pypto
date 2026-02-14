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
#include "tilefwk/tilefwk_log.h"

namespace npu::tile_fwk {
namespace {
const int FILE_AUTHORITY = 0640;
}

bool FileExist(const std::string &fPath) {
    return !RealPath(fPath).empty();
}

std::string RealPath(const std::string &path) {
    if (path.empty()) {
        FUNCTION_LOGI("path string is nullptr.");
        return "";
    }
    if (path.size() >= PATH_MAX) {
        FUNCTION_LOGI("file path %s is too long.", path.c_str());
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
        FUNCTION_LOGI("path %s is not exist.", path.c_str());
    }
    return res;
}

bool GetFileSize(const std::string& fPath, uint32_t &fileSize) {
    if (RealPath(fPath).empty()) {
        return false;
    }
    std::ifstream file(fPath, std::ios::binary | std::ios::ate); // 打开文件，定位到文件末尾
    if (!file.is_open()) {
        return false;
    }
    fileSize = file.tellg();
    file.close();
    return true;
}

uint32_t GetFileSize(const std::string& fPath) {
    uint32_t fileSize = 0;
    (void) GetFileSize(fPath, fileSize);
    return fileSize;
}

bool CreateDir(const std::string &directoryPath) {
    int32_t ret = mkdir(directoryPath.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);  // 755
    if (ret != 0 && errno != EEXIST) {
        FUNCTION_LOGW("Create dir[%s] failed, reason is %s", directoryPath.c_str(), strerror(errno));
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
            FUNCTION_LOGW("Create dir[%s] failed, reason is %s", directoryPath.c_str(), strerror(errno));
            return false;
        }
    }
    return true;
}

bool CreateMultiLevelDir(const std::string &directoryPath) {
    auto dirPathLen = directoryPath.length();
    if (dirPathLen >= PATH_MAX) {
        FUNCTION_LOGW("Path[%s] is too long, it must be less than %d", directoryPath.c_str(), PATH_MAX);
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

    ret = mkdir(directoryPath.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);  // 755
    if (ret != 0 && errno != EEXIST) {
        FUNCTION_LOGW("Create dir[%s] failed, reason is: %s", directoryPath.c_str(), strerror(errno));
        return false;
    }

    FUNCTION_LOGD("Create multi level dir [%s] successfully.", directoryPath.c_str());
    return true;
}

void DeleteFile(const std::string &path)
{
    if (path.empty()) {
        FUNCTION_LOGW("File name is empty.");
        return;
    }
    struct stat statBuf;
    if (lstat(path.c_str(), &statBuf) != 0) {
        FUNCTION_LOGW("Stat file[%s] failed.", path.c_str());
        return;
    }
    if (S_ISREG(statBuf.st_mode) == 0) {
        FUNCTION_LOGW("[%s] is not a file.", path.c_str());
        return;
    }
    int res = remove(path.c_str());
    if (res != 0) {
        FUNCTION_LOGW("Delete file[%s] failed.", path.c_str());
    }
}

bool ReadJsonFile(const std::string &file, nlohmann::json &jsonObj) {
    std::string path = RealPath(file);
    if (path.empty()) {
        FUNCTION_LOGW("File path [%s] does not exist", file.c_str());
        return false;
    }
    std::ifstream ifStream(path);
    try {
        if (!ifStream.is_open()) {
            FUNCTION_LOGW("Open %s failed, file is already open", file.c_str());
            return false;
        }

        ifStream >> jsonObj;
        ifStream.close();
    } catch (const std::exception &e) {
        FUNCTION_LOGW("Fail to convert file[%s] to Json. Exception message is .%s", path.c_str(), e.what());
        ifStream.close();
        return false;
    }

    return true;
}

bool ReadBytesFromFile(const std::string &fPath, std::vector<char> &buffer)
{
    std::string realPath = RealPath(fPath);
    if (realPath.empty()) {
        FUNCTION_LOGW("Bin file path[%s] is not valid.", fPath.c_str());
        return false;
    }

    std::ifstream ifStream(realPath.c_str(), std::ios::binary | std::ios::ate);
    if (!ifStream.is_open()) {
        FUNCTION_LOGW("read file %s failed.", fPath.c_str());
        return false;
    }
    try {
        std::streamsize size = ifStream.tellg();
        if (size <= 0 || size > INT_MAX) {
            ifStream.close();
            FUNCTION_LOGW("File size %ld is not within the range: (0, %d].", size, INT_MAX);
            return false;
        }

        ifStream.seekg(0, std::ios::beg);

        buffer.resize(size);
        ifStream.read(&buffer[0], size);
        FUNCTION_LOGD("Release file(%s) handle.", realPath.c_str());
        ifStream.close();
        FUNCTION_LOGD("Read size: %ld.", size);
    } catch (const std::ifstream::failure& e) {
        FUNCTION_LOGW("Fail to read file %s. Exception: %s.", fPath.c_str(), e.what());
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
        FUNCTION_LOGW("Open directory [%s] failed", path.c_str());
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

void SaveFile(const std::string &fPath, const std::vector<uint8_t> &data) {
    FILE *file = fopen(fPath.c_str(), "wb");
    if (file == nullptr) {
        FUNCTION_LOGW("Open file [%s] failed.", fPath.c_str());
        return;
    }
    fwrite(data.data(), 1, data.size(), file);
    fclose(file);
}

bool SaveFile(const std::string &fPath, const uint8_t *data, size_t size) {
    FILE *file = fopen(fPath.c_str(), "wb");
    if (file == nullptr) {
        FUNCTION_LOGW("Open file [%s] failed.", fPath.c_str());
        return false;
    }
    fwrite(data, 1, size, file);
    fclose(file);
    return true;
}

void SaveFileSafe(const std::string &fPath, const uint8_t *data, size_t size) {
    auto tmpfile = fPath + ".tmp";
    if (SaveFile(tmpfile, data, size)) {
        Rename(tmpfile, fPath);
    }
}

void Rename(const std::string &oldPath, const std::string &newPath) {
    if (rename(oldPath.c_str(), newPath.c_str()) != 0) {
        FUNCTION_LOGW("Rename file %s to %s failed.", oldPath.c_str(), newPath.c_str());
    }
}

bool DumpFile(const char *data, const size_t size, const std::string &fPath) {
    // dump bin file
    std::ofstream outFile(fPath, std::ios::binary);
    if (!outFile) {
        FUNCTION_LOGE("Failed open file %s.", fPath.c_str());
        return false;
    }
    outFile.write(data, size);
    outFile.close();
    FUNCTION_LOGI("Bin file[%s] has been dumped.", fPath.c_str());
    return true;
}

bool DumpFile(const std::vector<uint8_t> &data, const std::string &fPath) {
    return DumpFile(reinterpret_cast<const char *>(data.data()), data.size(), fPath);
}

bool DumpFile(const std::string &text, const std::string &fPath) {
    return DumpFile(text.data(), text.size(), fPath);
}

std::vector<uint8_t> LoadFile(const std::string &fPath) {
    std::vector<uint8_t> binary;
    std::string realPath = RealPath(fPath);
    if (realPath.empty()) {
        FUNCTION_LOGW("Bin file path[%s] is not valid.", fPath.c_str());
        return binary;
    }

    FILE *file = fopen(fPath.c_str(), "rb");
    if (file != nullptr) {
        fseek(file, 0, SEEK_END);
        int size = ftell(file);
        binary.resize(size);
        fseek(file, 0, SEEK_SET);
        size_t readSize = fread(binary.data(), 1, size, file);
        if (readSize != static_cast<size_t>(size)) {
            binary.clear();
        }
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
        FUNCTION_LOGW("Delete dir[%s] failed, reason is %d", directoryPath.c_str(), ret);
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
        FUNCTION_LOGW("Fail to lock file:", lockFilePath.c_str());
        fclose(fp);
        return nullptr;
    }
    FUNCTION_LOGI("Lock file successfully.", lockFilePath.c_str());
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
        FUNCTION_LOGW("Fail to open file:", srcPath.c_str(), ", ", dstPath.c_str());
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
        FUNCTION_LOGE("failed to call getcwd()");
        return "";
    }
    return cwd;
}

void RemoveOldestDirs(const std::string &path, const std::string &prefix, int left) {
    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        FUNCTION_LOGW("failed to opendir: %s", path.c_str());
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