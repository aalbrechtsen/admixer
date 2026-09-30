// admixer: everything printed goes to the screen and to the log file NAME.K.log.
#pragma once
#include <cstdarg>
#include <cstdio>

inline FILE*& log_file() {
  static FILE* fp = nullptr;
  return fp;
}

inline void say(const char* fmt, ...) {
  va_list a;
  va_start(a, fmt);
  std::vfprintf(stdout, fmt, a);
  va_end(a);
  std::fflush(stdout);
  if (FILE* fp = log_file()) {
    va_start(a, fmt);
    std::vfprintf(fp, fmt, a);
    va_end(a);
    std::fflush(fp);
  }
}

inline void say_error(const char* msg) {
  std::fprintf(stderr, "Error: %s\n", msg);
  if (FILE* fp = log_file()) std::fprintf(fp, "Error: %s\n", msg), std::fflush(fp);
}
