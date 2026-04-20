/*
 * Copyright (c) PyPTO Contributors.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
#include "bindings.h"

#include "ir/expr.h"
#include "core/error.h"

namespace pypto {

void BindError(py::module& m)
{
    // Register custom exception types and map them to Python exceptions
    static py::exception<ir::Error> exc_error(m, "Error", PyExc_Exception);
    static py::exception<ir::ValueError> exc_value_error(m, "ValueError", PyExc_ValueError);
    static py::exception<ir::TypeError> exc_type_error(m, "TypeError", PyExc_TypeError);
    static py::exception<ir::RuntimeError> exc_runtime_error(m, "RuntimeError", PyExc_RuntimeError);
    static py::exception<ir::NotImplementedError> exc_not_implemented_error(
        m, "NotImplementedError", PyExc_NotImplementedError);
    static py::exception<ir::IndexError> exc_index_error(m, "IndexError", PyExc_IndexError);
    static py::exception<ir::AssertionError> exc_assertion_error(m, "AssertionError", PyExc_AssertionError);
    static py::exception<ir::InternalError> exc_internal_error(m, "InternalError", PyExc_RuntimeError);

    // Set __module__ so the exception displays as "pypto.ir.InternalError"
    PyObject* internal_error_type = exc_internal_error.ptr();
    PyObject_SetAttrString(internal_error_type, "__module__", PyUnicode_FromString("pypto.ir"));

    // Register exception translator to convert C++ exceptions to Python exceptions
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const ir::ValueError& e) {
            PyErr_SetString(PyExc_ValueError, e.GetFullMessage().c_str());
        } catch (const ir::TypeError& e) {
            PyErr_SetString(PyExc_TypeError, e.GetFullMessage().c_str());
        } catch (const ir::RuntimeError& e) {
            PyErr_SetString(PyExc_RuntimeError, e.GetFullMessage().c_str());
        } catch (const ir::NotImplementedError& e) {
            PyErr_SetString(PyExc_NotImplementedError, e.GetFullMessage().c_str());
        } catch (const ir::IndexError& e) {
            PyErr_SetString(PyExc_IndexError, e.GetFullMessage().c_str());
        } catch (const ir::AssertionError& e) {
            PyErr_SetString(PyExc_AssertionError, e.GetFullMessage().c_str());
        } catch (const ir::InternalError& e) {
            PyErr_SetString(exc_internal_error.ptr(), e.GetFullMessage().c_str());
        } catch (const ir::Error& e) {
            PyErr_SetString(PyExc_Exception, e.GetFullMessage().c_str());
        }
    });
}

void BindLogging(py::module& m)
{
    py::native_enum<LogLevel>(m, "LogLevel", "enum.IntEnum", "Enumeration of available log levels")
        .value("DEBUG", LogLevel::DEBUG, "Detailed information for debugging")
        .value("INFO", LogLevel::INFO, "General informational messages")
        .value("WARN", LogLevel::WARN, "Warning messages for potentially harmful situations")
        .value("ERROR", LogLevel::ERROR, "Error messages for failures")
        .value("FATAL", LogLevel::FATAL, "Critical errors that may cause termination")
        .value("EVENT", LogLevel::EVENT, "Special events and milestones")
        .value("NONE", LogLevel::NONE, "Disable all logging")
        .export_values()
        .finalize(); // Export values to module scope for convenience

    // Bind LoggerManager functions
    m.def("set_log_level", &LoggerManager::SetLevel, py::arg("level"), "Set the log level threshold.");
    m.def("get_log_level", &LoggerManager::GetLevel, "Get the current log level threshold.");
    m.def(
        "log", [](LogLevel level, std::string message) { pypto::Logger(level, __LINE__) << message; }, py::arg("level"),
        py::arg("message"), "Log a message at the DEBUG level");
    m.def(
        "raise_error",
        [](const std::string& error_type, const std::string& message) {
            if (error_type == "ValueError") {
                throw ir::ValueError(message);
            } else if (error_type == "TypeError") {
                throw ir::TypeError(message);
            } else if (error_type == "RuntimeError") {
                throw ir::RuntimeError(message);
            } else if (error_type == "NotImplementedError") {
                throw ir::NotImplementedError(message);
            } else if (error_type == "IndexError") {
                throw ir::IndexError(message);
            } else if (error_type == "Error") {
                throw ir::Error(message);
            } else if (error_type == "AssertionError") {
                throw ir::AssertionError(message);
            } else if (error_type == "InternalError") {
                throw ir::InternalError(message);
            } else {
                throw ir::Error("Unknown error type");
            }
        },
        py::arg("error_type"), py::arg("message"), "Raise a Error from C++ for testing error handling");
}

void BindCore(py::module& m)
{
    BindError(m);
    BindLogging(m);
}
} // namespace pypto
