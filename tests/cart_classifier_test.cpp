#include "CartClassifier.hpp"
#include "Exceptions.hpp"
#include <iostream>
#include <cmath>

static int fail(const char* msg) {
    std::cout << "cart_classifier_test: FAIL: " << msg << "\n";
    return 1;
}

int main() {
    Matrix X;
    X.set_size(6, 1);
    X.data() = {1, 2, 3, 4, 5, 6};
    std::vector<double> y = {0, 0, 0, 1, 1, 1};

    CartClassifier m(3, 2, 1);
    m.fit(X, y);

    auto cls = m.classes();
    if (cls.size() != 2) return fail("classes size");
    if (cls[0] != 0 || cls[1] != 1) return fail("classes values");

    auto proba = m.predict_proba(X);
    if (proba.rows() != 6 || proba.cols() != 2) return fail("proba shape");
    for (std::size_t i = 0; i < proba.rows(); ++i) {
        double s = 0.0;
        for (std::size_t c = 0; c < proba.cols(); ++c) {
            double p = proba(i, c);
            if (p < 0.0 || p > 1.0) return fail("proba out of range");
            s += p;
        }
        if (std::fabs(s - 1.0) > 1e-9) return fail("proba sum != 1");
    }

    auto preds = m.predict(X);
    for (std::size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] != y[i]) return fail("predict wrong label");
    }

    // predict до fit
    {
        CartClassifier mm(3, 2, 1);
        try {
            mm.predict(X);
            return fail("predict before fit");
        } catch (const NotFitted&) {}
    }

    // max_depth = 0 → константа, самая частая метка
    {
        CartClassifier mm(0, 2, 1);
        mm.fit(X, y);
        auto p = mm.predict(X);
        for (double v : p) {
            if (v != 0.0 && v != 1.0) return fail("depth=0 label");
        }
    }

    std::cout << "cart_classifier_test: OK\n";
    return 0;
}