#pragma once
#define ENABLE_NVTX
#ifdef ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#else
// Stub implementations when NVTX is disabled
#define nvtxRangePushA(name)
#define nvtxRangePop()
#endif

namespace gs {

class NvtxRange {
public:
    explicit NvtxRange(const char* name) {
#ifdef ENABLE_NVTX
        nvtxRangePushA(name);
#endif
        (void)name;
    }

    ~NvtxRange() {
#ifdef ENABLE_NVTX
        nvtxRangePop();
#endif
    }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

} // namespace gs
