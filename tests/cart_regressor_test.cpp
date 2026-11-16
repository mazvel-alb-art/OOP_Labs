#include "CartRegressor.hpp"
#include "Exceptions.hpp"
#include <iostream>
#include <cmath>

static int fail(const char* msg) {
    std::cout << "cart_regressor_test: FAIL: " << msg << "\n";
    return 1;
}

int main() {
    Matrix X;
    X.set_size(8, 1);
    X.data() = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<double> y = {1, 1, 2, 2, 3, 3, 4, 4};

    // max_depth = 0 → без ограничения глубины
    {
        CartRegressor m(0, 2, 1);
        m.fit(X, y);
        if (m.node_count() <= 1) return fail("unlimited depth nodes");
        if (m.leaf_count() <= 1) return fail("unlimited depth leaves");
        if (m.actual_depth() == 0) return fail("unlimited depth depth");
        auto p = m.predict(X);
        if (p.size() != y.size()) return fail("unlimited depth prediction size");
        for (double v : p) {
            if (!std::isfinite(v)) return fail("unlimited depth prediction finite");
        }
    }

    // max_depth = 1 → пень
    {
        CartRegressor m(1, 2, 1);
        m.fit(X, y);
        if (m.node_count() != 3) return fail("depth=1 nodes");
        if (m.leaf_count() != 2) return fail("depth=1 leaves");
        if (m.actual_depth() != 1) return fail("depth=1 depth");
    }

    // predict до fit
    {
        CartRegressor m(3, 2, 1);
        try {
            m.predict(X);
            return fail("predict before fit");
        } catch (const NotFitted&) {}
    }

    // Разное число признаков
    {
        CartRegressor m(3, 2, 1);
        m.fit(X, y);
        Matrix bad;
        bad.set_size(1, 2);
        bad.data() = {0.0, 0.0};
        try {
            m.predict(bad);
            return fail("feature mismatch");
        } catch (const FeatureCountMismatch&) {}
    }

    std::cout << "cart_regressor_test: OK\n";
    return 0;
}
