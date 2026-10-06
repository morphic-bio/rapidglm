#include <fstream>
#include <iostream>
#include <rapidglm/rapidglm.hpp>
#include <sstream>
int main(int argc, char **argv) {
    rapidglm::Matrix design(6, 1, 1);
    rapidglm::Vector y{1, 2, 3, 2, 1, 3};
    rapidglm::FitOptions options;
    options.family = rapidglm::Family::poisson;
    if (argc == 3) {
        // Standalone consumer input: numeric design and one response vector.
        design = {};
        y.clear();
        std::ifstream file(argv[1]);
        std::string line;
        if (!file)
            return 2;
        while (std::getline(file, line)) {
            std::istringstream row(line);
            rapidglm::Vector values;
            double v;
            while (row >> v)
                values.push_back(v);
            if (!design.cols)
                design.cols = values.size();
            if (values.size() != design.cols || !row.eof())
                return 2;
            design.values.insert(design.values.end(), values.begin(), values.end());
            ++design.rows;
        }
        std::ifstream response(argv[2]);
        double value;
        while (response >> value)
            y.push_back(value);
        if (!response.eof())
            return 2;
        options.family = rapidglm::Family::negative_binomial;
    }
    auto result = rapidglm::fit(design, y, options);
    if (!result.ok()) {
        std::cerr << rapidglm::status_name(result.status) << '\n';
        return 1;
    }
    for (double b : result.coefficients)
        std::cout << b << ' ';
    std::cout << '\n';
}
