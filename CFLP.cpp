#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ctime>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

const std::string DEFAULT_INSTANCE_NAME = "cap41_ss.txt";

struct Problem {
    int facilities = 0;
    int customers = 0;
    std::vector<int> capacities;
    std::vector<double> opening_costs;
    std::vector<int> demands;
    std::vector<std::vector<double>> assignment_costs;
};

struct Solution {
    std::vector<bool> facility_open;
    std::vector<int> customer_assignment;
    std::vector<int> facility_loads;
    double objective_value = -1.0;
};

Problem load_problem(const std::string& path);
std::string name_from_path(const std::string& path);
void print_problem(const Problem& problem);
void update_solution_state(const Problem& problem, Solution& solution);
bool is_valid_solution(const Problem& problem, const Solution& solution);
double evaluate_solution(const Problem& problem, Solution& solution);
void print_solution(const Solution& solution);
void print_problem_checks(const Problem& problem);
Solution random_solution(const Problem& problem, std::mt19937& rng);
Solution greedy_solution(const Problem& problem);

int main(int argc, char* argv[]) {
    try {
        std::string path = DEFAULT_INSTANCE_NAME;
        if (argc > 1) {
            path = argv[1];
        }

        Problem problem = load_problem(path);
        std::mt19937 rng(static_cast<unsigned int>(std::time(nullptr)));
        Solution random = random_solution(problem, rng);
        Solution greedy = greedy_solution(problem);

        std::cout << "Instancja: " << name_from_path(path) << "\n\n";
        print_problem(problem);
        std::cout << '\n';
        print_problem_checks(problem);
        std::cout << "\nLosowe rozwiazanie:\n";
        print_solution(random);
        std::cout << "objective_value: " << random.objective_value << '\n';
        std::cout << "\nZachlanne rozwiazanie:\n";
        print_solution(greedy);
        std::cout << "objective_value: " << greedy.objective_value << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Blad: " << error.what() << '\n';
        return 1;
    }

    return 0;
}

Problem load_problem(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Nie udalo sie otworzyc " + path);
    }

    Problem problem;
    if (!(input >> problem.facilities >> problem.customers)) {
        throw std::runtime_error("Nie udalo sie odczytac liczby magazynow i klientow z pliku " + path);
    }

    problem.capacities.resize(problem.facilities);
    problem.opening_costs.resize(problem.facilities);

    for (int facility = 0; facility < problem.facilities; ++facility) {
        if (!(input >> problem.capacities[facility] >> problem.opening_costs[facility])) {
            throw std::runtime_error("Nie udalo sie odczytac danych magazynu nr " + std::to_string(facility));
        }
    }

    problem.demands.resize(problem.customers);
    problem.assignment_costs.assign(problem.customers, std::vector<double>(problem.facilities, 0.0));

    for (int customer = 0; customer < problem.customers; ++customer) {
        if (!(input >> problem.demands[customer])) {
            throw std::runtime_error("Nie udalo sie odczytac popytu klienta nr " + std::to_string(customer));
        }

        for (int facility = 0; facility < problem.facilities; ++facility) {
            if (!(input >> problem.assignment_costs[customer][facility])) {
                throw std::runtime_error(
                    "Nie udalo sie odczytac kosztu przypisania dla klienta nr " +
                    std::to_string(customer) + " i magazynu nr " + std::to_string(facility)
                );
            }
        }
    }

    return problem;
}

std::string name_from_path(const std::string& path) {
    std::size_t slash_pos = path.find_last_of("/\\");
    std::string file_name = (slash_pos == std::string::npos) ? path : path.substr(slash_pos + 1);

    std::size_t dot_pos = file_name.find_last_of('.');
    return file_name.substr(0, dot_pos);
}

void update_solution_state(const Problem& problem, Solution& solution) {
    solution.facility_loads.assign(problem.facilities, 0);
    solution.facility_open.assign(problem.facilities, false);

    for (int customer = 0; customer < static_cast<int>(solution.customer_assignment.size()); ++customer) {
        int facility = solution.customer_assignment[customer];
        if (facility < 0 || facility >= problem.facilities) {
            continue;
        }

        solution.facility_open[facility] = true;
        solution.facility_loads[facility] += problem.demands[customer];
    }
}

bool is_valid_solution(const Problem& problem, const Solution& solution) {
    if (static_cast<int>(solution.customer_assignment.size()) != problem.customers) {
        return false;
    }

    if (static_cast<int>(solution.facility_loads.size()) != problem.facilities) {
        return false;
    }

    if (static_cast<int>(solution.facility_open.size()) != problem.facilities) {
        return false;
    }

    for (int customer = 0; customer < problem.customers; ++customer) {
        int facility = solution.customer_assignment[customer];
        if (facility < 0 || facility >= problem.facilities) {
            return false;
        }

        if (!solution.facility_open[facility]) {
            return false;
        }
    }

    for (int facility = 0; facility < problem.facilities; ++facility) {
        if (solution.facility_loads[facility] > problem.capacities[facility]) {
            return false;
        }
    }

    return true;
}

double evaluate_solution(const Problem& problem, Solution& solution) {
    update_solution_state(problem, solution);

    if (!is_valid_solution(problem, solution)) {
        solution.objective_value = std::numeric_limits<double>::max();
        return solution.objective_value;
    }

    double total_cost = 0.0;

    for (int facility = 0; facility < problem.facilities; ++facility) {
        if (solution.facility_open[facility]) {
            total_cost += problem.opening_costs[facility];
        }
    }

    for (int customer = 0; customer < problem.customers; ++customer) {
        int facility = solution.customer_assignment[customer];
        total_cost += problem.assignment_costs[customer][facility];
    }

    solution.objective_value = total_cost;
    return total_cost;
}

void print_problem(const Problem& problem) {
    std::cout << "Liczba magazynow: " << problem.facilities << '\n';
    std::cout << "Liczba klientow:  " << problem.customers << "\n\n";

    std::cout << "Magazyny:\n";
    for (int facility = 0; facility < problem.facilities; ++facility) {
        std::cout
            << "M" << facility
            << " capacity=" << problem.capacities[facility]
            << " opening_cost=" << std::fixed << std::setprecision(2) << problem.opening_costs[facility]
            << '\n';
    }

    std::cout << "\nKlienci (podglad):\n";
    const int preview_customers = std::min(problem.customers, 5);
    for (int customer = 0; customer < preview_customers; ++customer) {
        std::cout << "K" << customer << " demand=" << problem.demands[customer] << " costs: ";

        for (int facility = 0; facility < problem.facilities; ++facility) {
            std::cout << problem.assignment_costs[customer][facility];
            if (facility + 1 < problem.facilities) {
                std::cout << ", ";
            }
        }

        std::cout << '\n';
    }

    if (problem.customers > preview_customers) {
        std::cout << "... (" << problem.customers - preview_customers << " kolejnych klientow pominieto w podgladzie)\n";
    }
}

void print_solution(const Solution& solution) {
    std::cout << "Otwarte magazyny: ";
    for (int facility = 0; facility < static_cast<int>(solution.facility_open.size()); ++facility) {
        if (solution.facility_open[facility]) {
            std::cout << facility << ' ';
        }
    }
    std::cout << '\n';

    std::cout << "Obciazenia magazynow: ";
    for (int facility = 0; facility < static_cast<int>(solution.facility_loads.size()); ++facility) {
        std::cout << "[" << facility << ": " << solution.facility_loads[facility] << "] ";
    }
    std::cout << '\n';

    std::cout << "Przypisania per magazyn:\n";
    for (int facility = 0; facility < static_cast<int>(solution.facility_open.size()); ++facility) {
        std::cout << "M" << facility << ": ";

        bool has_customer = false;
        for (int customer = 0; customer < static_cast<int>(solution.customer_assignment.size()); ++customer) {
            if (solution.customer_assignment[customer] == facility) {
                std::cout << "K" << customer << ' ';
                has_customer = true;
            }
        }

        if (!has_customer) {
            std::cout << "-";
        }
        std::cout << '\n';
    }
}

void print_problem_checks(const Problem& problem) {
    int total_capacity = 0;
    int total_demand = 0;
    int max_capacity = 0;

    for (int facility = 0; facility < problem.facilities; ++facility) {
        total_capacity += problem.capacities[facility];
        max_capacity = std::max(max_capacity, problem.capacities[facility]);
    }

    std::vector<int> oversized_customers;
    for (int customer = 0; customer < problem.customers; ++customer) {
        total_demand += problem.demands[customer];
        if (problem.demands[customer] > max_capacity) {
            oversized_customers.push_back(customer);
        }
    }

    std::cout << "Kontrola danych:\n";
    std::cout << "Suma pojemnosci: " << total_capacity << '\n';
    std::cout << "Suma popytow:    " << total_demand << '\n';
    std::cout << "Max capacity:    " << max_capacity << '\n';

    if (oversized_customers.empty()) {
        return;
    }

    std::cout << "Uwaga: sa klienci wieksi niz pojemnosc pojedynczego magazynu: ";
    for (int customer : oversized_customers) {
        std::cout << "K" << customer << "(" << problem.demands[customer] << ") ";
    }
    std::cout << '\n';
}

Solution random_solution(const Problem& problem, std::mt19937& rng) {
    const int max_attempts = 200;

    std::vector<int> base_customer_order(problem.customers);
    std::iota(base_customer_order.begin(), base_customer_order.end(), 0);

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        Solution solution;
        solution.customer_assignment.assign(problem.customers, -1);

        std::vector<int> remaining_capacity = problem.capacities;
        std::vector<int> customer_order = base_customer_order;
        std::shuffle(customer_order.begin(), customer_order.end(), rng);

        bool success = true;

        for (int customer : customer_order) {
            std::vector<int> feasible_facilities;

            for (int facility = 0; facility < problem.facilities; ++facility) {
                if (remaining_capacity[facility] >= problem.demands[customer]) {
                    feasible_facilities.push_back(facility);
                }
            }

            if (feasible_facilities.empty()) {
                success = false;
                break;
            }

            std::uniform_int_distribution<int> distribution(
                0,
                static_cast<int>(feasible_facilities.size()) - 1
            );
            int chosen_facility = feasible_facilities[distribution(rng)];

            solution.customer_assignment[customer] = chosen_facility;
            remaining_capacity[chosen_facility] -= problem.demands[customer];
        }

        if (!success) {
            continue;
        }

        solution.objective_value = evaluate_solution(problem, solution);
        if (is_valid_solution(problem, solution)) {
            return solution;
        }
    }

    throw std::runtime_error("Nie udalo sie wygenerowac losowego rozwiazania dopuszczalnego.");
}

Solution greedy_solution(const Problem& problem) {
    Solution solution;
    solution.customer_assignment.assign(problem.customers, -1);

    std::vector<int> remaining_capacity = problem.capacities;
    std::vector<bool> facility_open(problem.facilities, false);
    std::vector<int> customer_order(problem.customers);
    std::iota(customer_order.begin(), customer_order.end(), 0);

    std::stable_sort(
        customer_order.begin(),
        customer_order.end(),
        [&problem](int left_customer, int right_customer) {
            return problem.demands[left_customer] > problem.demands[right_customer];
        }
    );

    for (int customer : customer_order) {
        int best_facility = -1;
        double best_increment = std::numeric_limits<double>::max();

        for (int facility = 0; facility < problem.facilities; ++facility) {
            if (remaining_capacity[facility] < problem.demands[customer]) {
                continue;
            }

            double increment = problem.assignment_costs[customer][facility];
            if (!facility_open[facility]) {
                increment += problem.opening_costs[facility];
            }

            if (best_facility == -1 || increment < best_increment) {
                best_facility = facility;
                best_increment = increment;
            }
        }

        if (best_facility == -1) {
            throw std::runtime_error("Nie udalo sie wygenerowac zachlannego rozwiazania dopuszczalnego.");
        }

        solution.customer_assignment[customer] = best_facility;
        remaining_capacity[best_facility] -= problem.demands[customer];
        facility_open[best_facility] = true;
    }

    solution.objective_value = evaluate_solution(problem, solution);
    return solution;
}
