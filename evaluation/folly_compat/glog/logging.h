#pragma once

#include <cstdlib>
#include <iostream>

namespace google {

enum LogSeverity {
  GLOG_INFO,
  GLOG_WARNING,
  GLOG_ERROR,
  GLOG_FATAL,
  GLOG_DFATAL
};

class LogMessage {
 public:
  LogMessage(LogSeverity severity, const char* file, int line)
      : severity_(severity) {
    std::cerr << file << ':' << line << ": ";
  }

  ~LogMessage() {
    std::cerr << std::endl;
    if (severity_ == GLOG_FATAL || severity_ == GLOG_DFATAL) {
      std::abort();
    }
  }

  std::ostream& stream() {
    return std::cerr;
  }

 private:
  LogSeverity severity_;
};

class LogMessageFatal : public LogMessage {
 public:
  LogMessageFatal(const char* file, int line)
      : LogMessage(GLOG_FATAL, file, line) {}
};

inline void InitGoogleLogging(const char*) {}

} // namespace google

#define LOG(severity) ::google::LogMessage(::google::GLOG_##severity, __FILE__, __LINE__).stream()
#define PLOG(severity) LOG(severity)
#define VLOG(level) while (false) LOG(INFO)

#define CHECK(cond) while (!(cond)) ::google::LogMessageFatal(__FILE__, __LINE__).stream()
#define CHECK_EQ(a, b) CHECK((a) == (b))
#define CHECK_NE(a, b) CHECK((a) != (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))
#define CHECK_NOTNULL(x) (x)

#define DCHECK(cond) while (false) ::google::LogMessage(::google::GLOG_INFO, __FILE__, __LINE__).stream()
#define DCHECK_EQ(a, b) DCHECK(true)
#define DCHECK_NE(a, b) DCHECK(true)
#define DCHECK_LT(a, b) DCHECK(true)
#define DCHECK_LE(a, b) DCHECK(true)
#define DCHECK_GT(a, b) DCHECK(true)
#define DCHECK_GE(a, b) DCHECK(true)
