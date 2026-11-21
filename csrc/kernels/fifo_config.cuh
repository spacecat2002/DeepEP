#pragma once

#include <cstdlib>

namespace deep_ep {
// The sole production transport selector. Shape and dtype safety remain automatic.
inline bool optimized_fifo_enabled(int ranks) {
    const char* value = std::getenv("DEEPEP_FIFO_OPTIMIZED");
    return value && value[0] == '1' && value[1] == '\0' &&
           (ranks == 2 || ranks == 4 || ranks == 8);
}
}  // namespace deep_ep
