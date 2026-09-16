/************************************************************
* IBM Confidential
* (C) Copyright IBM Corp. 2018, 2025
************************************************************/

/*
 * Description: Dimensions for data-structures/ops in DeepTools
 * Owner:  Swagath
 *
 */

#ifndef SEN_DATA_DIMS
#define SEN_DATA_DIMS

#include <util/sendefs/sendefs.h>

#include <cmath>
#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <unordered_map>

// please do not include json11.hpp here, as it is internal to deeptools
namespace json11 {
  class Json;
}


/*
 * classification of dimension types
 */

enum PrimaryDimTypes {
  IN,
  OUT,
  IJ,
  MB,
  X,
  Y,
  KIJ,
  I,
  J,
  KI,
  KJ,
  X1,
  PrimaryDimTypesCount
};

enum class PadType {
  NOPAD,
  LOWERED_PADDED,    // window-pad dimensions only (conv/pooling)
  PADDED_NOZEROPAD,  // window-pad dimensions only (conv/pooling)
  PADDED_WZEROPAD,   // window-pad dimensions only (conv/pooling)
  PADDED_FULLSPAN,   // non-window-pad dimensions only (csq/qfp)
  PADDED_FULLSPAN_WUNNEEDED,
};

enum MetaDimKind {
  Unpadded,
  Padded,
  PadFront,
  PadBack,
  PadValid,
  WindowDim,
  Stride,
  Dilation,
  Count
};

namespace EnumsConversion {
extern const std::map<std::string, MetaDimKind> stringToMetaDimKind;
extern const std::map<MetaDimKind, std::string> metaDimKindToString;
}  // namespace EnumsConversion

struct PrimaryDimAndKind {
  PrimaryDimTypes dim_ = PrimaryDimTypes::PrimaryDimTypesCount;
  MetaDimKind kind_ = MetaDimKind::Unpadded;
  PrimaryDimAndKind(PrimaryDimTypes dim = PrimaryDimTypes::PrimaryDimTypesCount,
                    MetaDimKind kind = MetaDimKind::Unpadded)
      : dim_(dim), kind_(kind) {}
};

// make PrimaryDimAndKind hashable & comparable to use as key in unordered map
bool operator==(const PrimaryDimAndKind &first,
                const PrimaryDimAndKind &second);
template <>
struct std::hash<PrimaryDimAndKind> {
  std::size_t operator()(const PrimaryDimAndKind &k) const {
    return ((k.dim_ << int(std::log2(int(MetaDimKind::Count)) + 1)) ^ k.kind_);
  }
};

struct PaddingFormType {
 public:
  typedef std::map<PrimaryDimTypes, PadType> PerDimPaddingInfoT;
  using iterator = typename PerDimPaddingInfoT::iterator;
  using const_iterator = typename PerDimPaddingInfoT::const_iterator;

  PadType getPadding(PrimaryDimTypes dim) const;
  std::string getPaddingAsStr(PrimaryDimTypes dim) const;
  void setPadding(PrimaryDimTypes dim, PadType pad);
  PaddingFormType() = default;
  PaddingFormType(PrimaryDimTypes dim, PadType pad) { setPadding(dim, pad); }
  void clear() { padding_.clear(); }

  // Some scenarios require traversing the entire map. The following methods
  // are added to allow such traversal.
  // Example: JSON import/export operation.
  iterator begin() { return padding_.begin(); }
  iterator end() { return padding_.end(); }
  const_iterator begin() const { return padding_.begin(); }
  const_iterator end() const { return padding_.end(); }
  bool hasPaddingInfo() const { return !padding_.empty(); }

  void print(std::ostream &out, int indent = 0) const;

 private:
  PerDimPaddingInfoT padding_;
};

namespace EnumsConversion {
extern const std::map<PrimaryDimTypes, std::string> primaryDimToString;
extern const std::map<std::string, PrimaryDimTypes> stringToPrimaryDim;
extern const std::map<PadType, std::string> padTypeToString;
extern const std::map<std::string, PadType> stringToPadType;
}  // namespace EnumsConversion

template <>
inline PrimaryDimTypes FromString(const std::string &s) {
  return EnumsConversion::stringToPrimaryDim.at(s);
}

struct DimPaddingSizes {
  int padFront_ = 0;
  int padBack_ = 0;
  int unneededPad_ = 0;       // total unneeded elements
  int unneededPadFront_ = 0;  // unneeded elements that come from padFront
  int unneededPadBack_ = 0;   // unneeded elements that come from padBack
  int stride_ = 1;
  int dilation_ = 1;
  PrimaryDimTypes windowDim_ = PrimaryDimTypes::PrimaryDimTypesCount;

  const int &getMetaDimVal(MetaDimKind kind) const;
  bool operator==(const DimPaddingSizes &A) const;
};

struct SymbolicDimInfo {
  int maxSize_ = -1;
  int granularity_ = -1;

  bool operator==(const SymbolicDimInfo &A) const {
    return maxSize_ == A.maxSize_ && granularity_ == A.granularity_;
  }
};

enum SenComponents : int;
class DataStructDims {
 public:
  std::string name_;
  // Primary dims
  double in_ = -1;   // input features   (or channels)
  double out_ = -1;  // output features (or channels)
  double mb_ = -1;   // minibatch size
  double ij_ = -1;   // output image dimensions (rows/cols)
  double rc_ = -1;   // input image (rows/cols) with zero padding
  double kij_ = -1;  // kernel dimensions (rows/cols)
  double y_ = -1;    // a kernel reuse dimension (e.g. timestep)
  double x_ =
      -1;  // a repeat dim that does not add reuse (e.g. attention heads)
  double x1_ =
      -1;  // a repeat dim that does not add reuse (e.g. attention heads)

  // to be removed in future..
  double sij_ = -1;  // stride dimensions (rows/cols)
  double zij_ = -1;  // zero pad dimensions (rows/cols)

  // Derived Dims
  double i_ = -1;   // output image rows
  double j_ = -1;   // output image cols
  double r_ = -1;   // input image rows with zero padding
  double c_ = -1;   // input image cols with zero padding
  double ki_ = -1;  // kernel rows
  double kj_ = -1;  // kernel cols
  double si_ = -1;  // stride along rows
  double sj_ = -1;  // stride along cols
  // Zi and Zj denote zero padding at each side..
  // Zi = top and bottom and Zj = left and right
  // if integer, i.e. 'x', then both left/top or bottom/right are padded by 'x'
  // if fraction, i.e. 'x.5', then left/top padding is floor(x.5) = 'x' and
  // right/bottom padding is ceil(x.5) = 'x+1'
  double zi_ = -1;  // zero pad rows
  double zj_ = -1;  // zero pad cols

  // for each symbolic dimension, specify max and granularity sizes. When dim is
  // symbolic, set main dim size in DataStructDims to max
  std::map<PrimaryDimTypes, SymbolicDimInfo> symbolicDimInfo_;

  // when multiple symbolic dims are present, it is possible to also set a max
  // symbolic volume, usually smaller than the multiplication of the max of all
  // the symbolic dims involved
  std::map<std::set<PrimaryDimTypes>, int> maxSymbolicVolume_;

  // for each dimension where work is split, a vector will explicitly show the
  // amount of work for each corelet
  std::map<PrimaryDimTypes, std::vector<int>> coreletSplit_;
  // for each dimension where work is split, for each coreletid, a vector will
  // explicitly show the amount of work for each pt row
  std::map<PrimaryDimTypes, std::map<int, std::vector<int>>> rowSplit_;
  // for each dimension where work is split, for each coreletid, a vector will
  // explicitly show the amount of work for PE and SFP for each corelet
  std::map<PrimaryDimTypes,
           std::map<int, std::unordered_map<SenComponents, int>>>
      peSfpSplit_;

  // for each primary dimension that has a padded version, include related
  // information that contribute to its padded size, like padding, stride,
  // associated window dim etc.
  std::map<PrimaryDimTypes, DimPaddingSizes> paddingSizes_;

  inline const auto tie() const {
    return std::tie(in_, out_, mb_, ij_, rc_, kij_, y_, x_, x1_, sij_, zij_, i_,
                    j_, r_, c_, ki_, kj_, si_, sj_, zi_, zj_, symbolicDimInfo_,
                    maxSymbolicVolume_, coreletSplit_, rowSplit_, peSfpSplit_,
                    paddingSizes_);
  }

  // default constructor: don't put anything
  DataStructDims() = default;

  void clear();
  /*
   * compute some of the primary dimensions
   * from the derived dimensions
   */
  void compound();

  /*
   * tells if the information has not been filled in
   */

  bool empty() const;

  /*
   * serialization/deserialization routines.
   * WARNING: do not modify them without DGP approval,
   *  for debugging purposes use the print() routine instead.
   */

  void write(std::ostream &out) const;

  void read(std::istream &str, double &var);
  void read(std::istream &in);

  /*
   *  Human friend print routine. This routine can be modified
   */

  friend std::ostream &operator<<(std::ostream &out, const DataStructDims &d);

  void printMed(std::ostream &out) const;
  void printShort(std::ostream &out) const;

  void exportJson(std::ostream &json, bool skipDeprecatedFields = false) const;
  bool importJsonObj(const json11::Json &json);

  double &paramNameToVal(const std::string &s);
  int primaryDimToVal_st(PrimaryDimTypes d) const;
  int primaryDimToVal_st(PrimaryDimTypes d, SenComponents peOrSfp,
                         int ptrowId = -1, int clId = -1,
                         const PaddingFormType &padded = {},
                         double dimDensity = 1.0,
                         bool getSymbolicGranularity = false) const;

  double &primaryDimToValHandler_st(PrimaryDimTypes d);

  int dataStageDimToVal_compView_st(PrimaryDimTypes d, SenComponents comp,
                                    int clId = -1,
                                    const PaddingFormType &padded = {},
                                    double dimDensity = 1.0,
                                    bool getSymbolicGranularity = false) const;

  bool operator==(const DataStructDims &A) const { return tie() == A.tie(); }

  void pruneMaxSymbolicVolumes(const DataStructDims &refDstg);
  void makeDimSymbolic(const DataStructDims &refDs, PrimaryDimTypes dim);
  void makeDimNotSymbolic(PrimaryDimTypes dim);

  int calculate_padded(PrimaryDimTypes d, int val,
                       const PaddingFormType &padding,
                       bool getSymbolicGranularity) const;

 private:
  int primaryDimToVal_base_st(PrimaryDimTypes d, const PaddingFormType &padded,
                              double dimDensity,
                              bool getSymbolicGranularity) const;
  int primaryDimToVal_clView_st(PrimaryDimTypes d, int clId,
                                const PaddingFormType &padded,
                                double dimDensity,
                                bool getSymbolicGranularity) const;

  int scaleFromMaxToGranularity(PrimaryDimTypes d, int val) const;
};

#endif
