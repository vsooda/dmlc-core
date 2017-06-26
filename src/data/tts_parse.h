//
// Created by sooda on 17-6-22.
//

#ifndef MXNET_TTS_PARSE_H
#define MXNET_TTS_PARSE_H
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/common.h>
#include <dmlc/parameter.h>
#include <dmlc/data.h>
#include <fstream>
#include <string>
#include <cstring>
#include "../io/filesys.h"

namespace dmlc {
namespace data {
inline std::string getFileString(const std::string &filepath) {
  std::ifstream is(filepath);
  std::string filebuffer = "";
  if (is.is_open()) {
    // get length of file:
    is.seekg(0, is.end);
    long long length = is.tellg();
    is.seekg(0, is.beg);
    char *buffer = new char[length];
    //std::cout << "Reading " << filepath << " " << length << " characters... ";
    is.read(buffer, length);
    is.close();
    // ...buffer contains the entire file...
    filebuffer = std::string(buffer, length);
    delete[] buffer;
  } else {
    std::cout << filepath << "open faild in getFileString" << std::endl;
  }
  return filebuffer;
}

inline std::string StripEnd(std::string str, char ch) {
  while (str.length() != 0 && str[str.length() - 1] == ch) {
    str.resize(str.length() - 1);
  }
  return str;
}

struct TTSParserParam : public dmlc::Parameter<TTSParserParam> {
  int feat_dims;
  std::string scp_name;
  DMLC_DECLARE_PARAMETER(TTSParserParam) {
    DMLC_DECLARE_FIELD(feat_dims).set_default(0)
        .describe("feature dim of one row");
    DMLC_DECLARE_FIELD(scp_name).set_default("NULL")
        .describe("file list name.");
  }
};


class TTSParser {
public:
  explicit TTSParser(std::string path_name, int feat_dims) {
    param_.scp_name = path_name;
    param_.feat_dims = feat_dims;
    Init();
  }

  void BeforeFirst();

  bool Next();

  real_t *Value() const;

  void Init();

  void LoadOneFile(std::string filename);

private:
  TTSParserParam param_;
  int current_index_;
  std::vector<real_t> rows_data_;
  std::vector<std::string> file_names_;
  int file_index_;
  real_t *out_;
  int frame_nums_;
};

inline void TTSParser::Init() {
  dmlc::io::URI path(param_.scp_name.c_str());
  size_t pos = path.name.rfind('/');
  if (pos == std::string::npos || pos + 1 == path.name.length()) {
    dmlc::io::URI dir = path;
    dir.name = path.name.substr(0, pos);
    std::vector<dmlc::io::FileInfo> dfiles;
    dmlc::io::FileSystem::GetInstance(path)->ListDirectory(dir, &dfiles);
    for (size_t i = 0; i < dfiles.size(); ++i) {
      file_names_.push_back(dfiles[i].path.name);
    }
  } else {
    pos = path.name.find(';');
    if (pos == std::string::npos) { //没有分号，说明是scp文件
      FILE *fin = fopen(param_.scp_name.c_str(), "r");
      if (fin == NULL) {
        std::cout << "open fail " << param_.scp_name << std::endl;
      }
      char file_name[800];
      memset(file_name, 0, sizeof(file_name));
      while (fscanf(fin, "%s%*c", file_name) != EOF) {
        std::cout << file_name << std::endl;
        file_names_.push_back(file_name);
      }
    } else {
      const char dlm = ';';
      file_names_ = Split(param_.scp_name, dlm);
    }
  }
  file_index_ = 0;
  current_index_ = 0;
  frame_nums_ = 0;
}


inline real_t *TTSParser::Value() const {
  return out_;
}

inline void TTSParser::LoadOneFile(std::string filename) {
  std::string file_str = getFileString(filename);
  int len = file_str.length() / sizeof(real_t);
  rows_data_.clear();
  rows_data_.resize(len);
  frame_nums_ = len / param_.feat_dims;
  current_index_ = 0;
  memcpy(rows_data_.data(), file_str.c_str(), sizeof(float) * frame_nums_ * param_.feat_dims);
}

inline bool TTSParser::Next() {
  if (current_index_ < frame_nums_) {
    out_ = rows_data_.data() + current_index_ * param_.feat_dims;
    current_index_++;
    return true;
  } else {
    if (file_index_ < file_names_.size()) {
      LoadOneFile(file_names_[file_index_++]);
      out_ = rows_data_.data() + current_index_ * param_.feat_dims;
      current_index_++;
      return true;
    } else {
      return false;
    }
  }
}

inline void TTSParser::BeforeFirst() {
  file_index_ = 0;
  current_index_ = 0;
  frame_nums_ = 0;
}

}
}

#endif //MXNET_TTS_PARSE_H
