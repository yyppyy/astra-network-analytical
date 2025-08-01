/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/Mesh.h"
#include <cassert>
#include <cmath>
#include <utility>

using namespace NetworkAnalyticalCongestionAware;

namespace {

// Choose rows, cols so rows*cols >= n and |rows-cols| is small (square-ish)
static std::pair<int,int> squareish_grid(int n) {
    int rows = static_cast<int>(std::floor(std::sqrt(static_cast<double>(n))));
    if (rows < 1) rows = 1;
    int cols = (n + rows - 1) / rows; // ceil(n / rows)
    // Ensure coverage in pathological cases
    while (rows * cols < n) ++cols;
    return {rows, cols};
}

// Row-major index
static inline int idx(int r, int c, int cols) { return r * cols + c; }

} // namespace

Mesh::Mesh(const int npus_count,
           const Bandwidth bandwidth,
           const Latency latency) noexcept
    : BasicTopology(npus_count, npus_count, bandwidth, latency) {
    assert(npus_count > 0);
    assert(bandwidth > 0);
    assert(latency >= 0);

    auto [rows, cols] = squareish_grid(npus_count);
    x_ = rows;
    y_ = cols;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const int u = idx(r, c, cols);
            if (u >= npus_count) continue; // cell beyond last NPU

            // Connect to right neighbor (no wrap)
            if (c + 1 < cols) {
                const int v = idx(r, c + 1, cols);
                if (v < npus_count) {
                    connect(u, v, bandwidth, latency, true);
                }
            }
            // Connect to down neighbor (no wrap)
            if (r + 1 < rows) {
                const int v = idx(r + 1, c, cols);
                if (v < npus_count) {
                    connect(u, v, bandwidth, latency, true);
                }
            }
        }
    }
}

Route Mesh::route(DeviceId src, DeviceId dest) const noexcept {
    // sanity
    assert(0 <= src && src < npus_count);
    assert(0 <= dest && dest < npus_count);
    assert(x_ > 0 && y_ > 0 && x_ * y_ >= npus_count);

    Route route;

    if (src == dest) {
        route.push_back(devices[dest]);
        return route;
    }

    auto id_of = [this](int row, int col) -> DeviceId {
        return static_cast<DeviceId>(row * y_ + col);
    };

    // linear id -> (row = x, col = y)
    int sx = static_cast<int>(src / y_);   // source row
    int sy = static_cast<int>(src % y_);   // source col
    const int dx = static_cast<int>(dest / y_); // dest row
    const int dy = static_cast<int>(dest % y_); // dest col

    DeviceId current = src;

    // --- X phase: move along rows until sx == dx ---
    if (sx != dx) {
        const int step_x = (dx > sx) ? 1 : -1;
        while (sx != dx) {
            route.push_back(devices[current]);
            sx += step_x;
            current = id_of(sx, sy);
        }
    }

    // --- Y phase: move along cols until sy == dy ---
    if (sy != dy) {
        const int step_y = (dy > sy) ? 1 : -1;
        while (sy != dy) {
            route.push_back(devices[current]);
            sy += step_y;
            current = id_of(sx, sy);
        }
    }

    // arrive at dest
    route.push_back(devices[dest]);
    return route;
}
