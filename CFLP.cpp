#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

const std::filesystem::path PROJECT_DIR = std::filesystem::path(__FILE__).parent_path();
const std::string DEFAULT_INSTANCE_NAME = "cap41_ss.txt";

const int RANDOM_RUNS = 10000;
const int EA_RUNS = 10;
const int SA_RUNS = 10;
const int VNS_RUNS = 10;
const int GRASP_RUNS = 10;

const std::string SUMMARY_CSV_PATH = (PROJECT_DIR / "summary.csv").string();
std::string EA_RUNS_DIR   = (PROJECT_DIR / "ea_runs4").string();
std::string SA_RUNS_DIR   = (PROJECT_DIR / "sa_runs").string();
std::string VNS_RUNS_DIR  = (PROJECT_DIR / "vns_runs").string();
std::string GRASP_RUNS_DIR = (PROJECT_DIR / "grasp_runs").string();

const bool SAVE_PROGRESS_CSV = true;
const bool EA_VERBOSE = false;
const bool DETERMINISTIC_REPAIR_TARGET = true;

struct EAConfig {
    int pop_size = 40;
    int gen = 250;
    int tour_size = 7;
    double mutation_pro = 0.90;
    double cross_pro = 0.85;
    double better_parent_bias = 0.50;
};

struct SAConfig {
    double initial_temp = 180000.0;
    double cooling_rate = 0.995;
    double min_temp = 8.0;
    int iter_per_temp = 5;
};

struct VNSConfig {
    int max_iterations = 50;
    int shake_attempts_per_neighborhood = 5;
    int local_search_attempts = 60;
};

struct GRASPConfig {
    int iterations = 100;
    int rcl_size = 2;
    int local_search_attempts = 100;
};

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
    std::vector<int> facility_customer_counts;
    double objective_value = -1.0;
};


struct Stats {
    double best = 0.0;
    double worst = 0.0;
    double avg = 0.0;
    double std = 0.0;
};

Problem load_problem(const std::string& path);
std::string name_from_path(const std::string& path);
void print_problem(const Problem& problem);
void update_solution_state(const Problem& problem, Solution& solution);
void move_customer_inplace(const Problem& problem, Solution& solution, int customer, int new_facility);
double evaluate_solution(const Problem& problem, Solution& solution);
Stats calculate_stats(const std::vector<double>& values);
std::vector<double> population_scores(const std::vector<Solution>& population);
void print_solution(const std::string& name, const Solution& solution);
void print_stats(const std::string& name, const Stats& stats);
void summary_row(
    const std::string& csv_path,
    const std::string& instance_name,
    const std::string& method_name,
    int runs,
    double best,
    std::optional<double> worst,
    std::optional<double> avg,
    std::optional<double> std,
    long long time_ms
);
std::string make_ea_csv_path(const std::string& instance_name, const EAConfig& config, int run_number);
void create_ea_csv(const std::string& csv_path);
void ea_progress_row(std::ofstream& csv_output, int generation, const Stats& stats);
std::string make_sa_csv_path(const std::string& instance_name, const SAConfig& config, int run_number);
void create_sa_csv(const std::string& csv_path);
void sa_progress_row(std::ofstream& csv_output, int iteration, double current_value, double best_value, double temperature);
std::string make_vns_csv_path(const std::string& instance_name, const VNSConfig& config, int run_number);
void create_vns_csv(const std::string& csv_path);
void vns_progress_row(std::ofstream& csv_output, int iteration, double shaken_value, double after_local_search_value, double best_value);
std::string make_grasp_csv_path(const std::string& instance_name, const GRASPConfig& config, int run_number);
void create_grasp_csv(const std::string& csv_path);
void grasp_progress_row(std::ofstream& csv_output, int iteration, double constructed_value, double improved_value, double best_value);
Solution random_solution(const Problem& problem, std::mt19937& rng);
Solution greedy_solution(const Problem& problem);
std::vector<Solution> initialize_population(const Problem& problem, const EAConfig& config, std::mt19937& rng);
int tournament_selection(const std::vector<Solution>& population, int tournament_size, std::mt19937& rng);
Solution evolutionary_algorithm(const Problem& problem, std::ofstream& csv_output, const EAConfig& config, std::mt19937& rng);
Solution sa_neighbor(const Problem& problem, const Solution& current, std::mt19937& rng);
Solution simulated_annealing(const Problem& problem, std::ofstream& csv_output, const SAConfig& config, std::mt19937& rng);
bool repair_solution(const Problem& problem, Solution& solution, std::mt19937& rng);
Solution local_search_move_one(
    const Problem& problem,
    const Solution& start,
    int max_attempts,
    int neighborhood,
    int& log_iteration,
    double& best_value,
    std::ofstream& csv_output,
    std::mt19937& rng
);
bool shake_swap_two(const Problem& problem, Solution& solution, std::mt19937& rng);
bool shake_move_two(const Problem& problem, Solution& solution, std::mt19937& rng);
bool shake_partial_facility_reallocation(const Problem& problem, Solution& solution, std::mt19937& rng);
Solution variable_neighborhood_search(const Problem& problem, std::ofstream& csv_output, const VNSConfig& config, std::mt19937& rng);
Solution grasp_construct_solution(const Problem& problem, const GRASPConfig& config, std::mt19937& rng);
Solution grasp_algorithm(const Problem& problem, std::ofstream& csv_output, const GRASPConfig& config, std::mt19937& rng);
void mutate_customer(const Problem& problem, Solution& solution, int customer, std::mt19937& rng);
void mutate_solution(const Problem& problem, Solution& solution, std::mt19937& rng, double mutation_probability);
Solution crossover(
    const Problem& problem,
    const Solution& parent1,
    const Solution& parent2,
    double better_parent_bias,
    std::mt19937& rng
);
void run_batch_experiments();
void run_tuning();

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--batch") {
        run_batch_experiments();
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--tune") {
        run_tuning();
        return 0;
    }
    std::string path = (PROJECT_DIR / DEFAULT_INSTANCE_NAME).string();
    if (argc > 1) {
        path = argv[1];
    }

    Problem problem = load_problem(path);
    std::string instance_name = name_from_path(path);
    std::mt19937 rng(std::random_device{}());
    std::cout << "Instancja: " << instance_name << "\n\n";
    print_problem(problem);
    std::cout << '\n';
    std::cout << "Wybor algorytmu:\n";
    std::cout << "1. algorytm losowy\n";
    std::cout << "2. algorytm zachlanny\n";
    std::cout << "3. algorytm ewolucyjny\n";
    std::cout << "4. algorytm symulowanego wyzarzania\n";
    std::cout << "5. algorytm VNS\n";
    std::cout << "6. algorytm GRASP\n";
    std::cout << "wybor: ";

    int method = 0;
    std::cin >> method;
    std::cout << '\n';

    switch (method) {
        case 1: {
            auto start_time = std::chrono::steady_clock::now();
            std::vector<double> results;
            results.reserve(RANDOM_RUNS);
            Solution best_random;
            bool has_best_solution = false;

            for (int run = 0; run < RANDOM_RUNS; ++run) {
                Solution current_solution = random_solution(problem, rng);
                results.push_back(current_solution.objective_value);

                if (!has_best_solution || current_solution.objective_value < best_random.objective_value) {
                    best_random = current_solution;
                    has_best_solution = true;
                }
            }

            Stats stats = calculate_stats(results);
            print_stats("Wyniki algorytmu losowego:", stats);
            std::cout << "Liczba uruchomien: " << RANDOM_RUNS << '\n';
            std::cout << '\n';
            print_solution("Najlepsze rozwiazanie:", best_random);
            auto end_time = std::chrono::steady_clock::now();
            long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            summary_row(
                SUMMARY_CSV_PATH,
                instance_name,
                "random",
                RANDOM_RUNS,
                stats.best,
                stats.worst,
                stats.avg,
                stats.std,
                elapsed_ms
            );
            return 0;
        }

        case 2: {
            auto start_time = std::chrono::steady_clock::now();
            Solution greedy = greedy_solution(problem);
            print_solution("Rozwiazanie zachlanne:", greedy);
            auto end_time = std::chrono::steady_clock::now();
            long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            summary_row(
                SUMMARY_CSV_PATH,
                instance_name,
                "greedy",
                1,
                greedy.objective_value,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                elapsed_ms
            );
            return 0;
        }

        case 3: {
            auto start_time = std::chrono::steady_clock::now();
            const EAConfig ea_config;
            std::vector<double> results;
            results.reserve(EA_RUNS);
            Solution best_overall;
            bool has_best_solution = false;

            std::cout << "EA config:\n";
            std::cout << "pop_size: " << ea_config.pop_size << '\n';
            std::cout << "gen: " << ea_config.gen << '\n';
            std::cout << "Px: " << ea_config.cross_pro << '\n';
            std::cout << "Pm: " << ea_config.mutation_pro << '\n';
            std::cout << "Tour: " << ea_config.tour_size << '\n';
            std::cout << "EA runs: " << EA_RUNS << "\n\n";

            for (int run = 1; run <= EA_RUNS; ++run) {
                std::ofstream ea_progress_output;
                if (SAVE_PROGRESS_CSV) {
                    std::string ea_progress_csv_path = make_ea_csv_path(instance_name, ea_config, run);
                    create_ea_csv(ea_progress_csv_path);
                    ea_progress_output.open(ea_progress_csv_path, std::ios::app);
                    if (!ea_progress_output) {
                        throw std::runtime_error("Nie udalo sie otworzyc pliku csv: " + ea_progress_csv_path);
                    }
                }

                std::cout << "========== RUN " << run << " ===========\n\n";

                Solution best = evolutionary_algorithm(problem, ea_progress_output, ea_config, rng);
                results.push_back(best.objective_value);

                if (!has_best_solution || best.objective_value < best_overall.objective_value) {
                    best_overall = best;
                    has_best_solution = true;
                }

                print_solution("Najlepszy osobnik EA:", best);
                std::cout << '\n';
            }

            Stats stats = calculate_stats(results);
            print_stats("Podsumowanie EA:", stats);
            std::cout << '\n';

            auto end_time = std::chrono::steady_clock::now();
            long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            summary_row(
                SUMMARY_CSV_PATH,
                instance_name,
                "EA",
                EA_RUNS,
                stats.best,
                stats.worst,
                stats.avg,
                stats.std,
                elapsed_ms
            );
            print_solution("Najlepszy osobnik EA ze wszystkich runow:", best_overall);
            return 0;
        }

        case 4: {
            auto start_time = std::chrono::steady_clock::now();
            const SAConfig sa_config;
            std::vector<double> results;
            results.reserve(SA_RUNS);
            Solution best_overall;
            bool has_best_solution = false;

            std::cout << "SA config:\n";
            std::cout << "T0: " << sa_config.initial_temp << '\n';
            std::cout << "cooling_rate: " << sa_config.cooling_rate << '\n';
            std::cout << "min_temp: " << sa_config.min_temp << '\n';
            std::cout << "iter_per_temp: " << sa_config.iter_per_temp << '\n';
            std::cout << "SA runs: " << SA_RUNS << "\n\n";

            for (int run = 1; run <= SA_RUNS; ++run) {
                std::ofstream sa_progress_output;
                if (SAVE_PROGRESS_CSV) {
                    std::string sa_progress_csv_path = make_sa_csv_path(instance_name, sa_config, run);
                    create_sa_csv(sa_progress_csv_path);
                    sa_progress_output.open(sa_progress_csv_path, std::ios::app);
                    if (!sa_progress_output) {
                        throw std::runtime_error("Nie udalo sie otworzyc pliku csv: " + sa_progress_csv_path);
                    }
                }

                std::cout << "========== RUN " << run << " ===========\n\n";

                Solution best = simulated_annealing(problem, sa_progress_output, sa_config, rng);
                results.push_back(best.objective_value);

                if (!has_best_solution || best.objective_value < best_overall.objective_value) {
                    best_overall = best;
                    has_best_solution = true;
                }

                print_solution("Najlepsze rozwiazanie SA:", best);
                std::cout << '\n';
            }

            Stats stats = calculate_stats(results);
            print_stats("Podsumowanie SA:", stats);
            std::cout << '\n';

            auto end_time = std::chrono::steady_clock::now();
            long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            summary_row(
                SUMMARY_CSV_PATH,
                instance_name,
                "SA",
                SA_RUNS,
                stats.best,
                stats.worst,
                stats.avg,
                stats.std,
                elapsed_ms
            );
            print_solution("Najlepsze rozwiazanie SA ze wszystkich runow:", best_overall);
            return 0;
        }

        case 5: {
            auto start_time = std::chrono::steady_clock::now();
            const VNSConfig vns_config;
            std::vector<double> results;
            results.reserve(VNS_RUNS);
            Solution best_overall;
            bool has_best_solution = false;

            std::cout << "VNS config:\n";
            std::cout << "max_iterations: " << vns_config.max_iterations << '\n';
            std::cout << "shake_attempts_per_neighborhood: " << vns_config.shake_attempts_per_neighborhood << '\n';
            std::cout << "local_search_attempts: " << vns_config.local_search_attempts << '\n';
            std::cout << "VNS runs: " << VNS_RUNS << "\n\n";

            for (int run = 1; run <= VNS_RUNS; ++run) {
                std::ofstream vns_progress_output;
                if (SAVE_PROGRESS_CSV) {
                    std::string vns_progress_csv_path = make_vns_csv_path(instance_name, vns_config, run);
                    create_vns_csv(vns_progress_csv_path);
                    vns_progress_output.open(vns_progress_csv_path, std::ios::app);
                    if (!vns_progress_output) {
                        throw std::runtime_error("Nie udalo sie otworzyc pliku csv: " + vns_progress_csv_path);
                    }
                }

                std::cout << "========== RUN " << run << " ===========\n\n";

                Solution best = variable_neighborhood_search(problem, vns_progress_output, vns_config, rng);
                results.push_back(best.objective_value);

                if (!has_best_solution || best.objective_value < best_overall.objective_value) {
                    best_overall = best;
                    has_best_solution = true;
                }

                print_solution("Najlepsze rozwiazanie VNS:", best);
                std::cout << '\n';
            }

            Stats stats = calculate_stats(results);
            print_stats("Podsumowanie VNS:", stats);
            std::cout << '\n';

            auto end_time = std::chrono::steady_clock::now();
            long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            summary_row(
                SUMMARY_CSV_PATH,
                instance_name,
                "VNS",
                VNS_RUNS,
                stats.best,
                stats.worst,
                stats.avg,
                stats.std,
                elapsed_ms
            );
            print_solution("Najlepsze rozwiazanie VNS ze wszystkich runow:", best_overall);
            return 0;
        }

        case 6: {
            auto start_time = std::chrono::steady_clock::now();
            const GRASPConfig grasp_config;
            std::vector<double> results;
            results.reserve(GRASP_RUNS);
            Solution best_overall;
            bool has_best_solution = false;

            std::cout << "GRASP config:\n";
            std::cout << "iterations: " << grasp_config.iterations << '\n';
            std::cout << "rcl_size: " << grasp_config.rcl_size << '\n';
            std::cout << "local_search_attempts: " << grasp_config.local_search_attempts << '\n';
            std::cout << "GRASP runs: " << GRASP_RUNS << "\n\n";

            for (int run = 1; run <= GRASP_RUNS; ++run) {
                std::ofstream grasp_progress_output;
                if (SAVE_PROGRESS_CSV) {
                    std::string grasp_progress_csv_path = make_grasp_csv_path(instance_name, grasp_config, run);
                    create_grasp_csv(grasp_progress_csv_path);
                    grasp_progress_output.open(grasp_progress_csv_path, std::ios::app);
                    if (!grasp_progress_output) {
                        throw std::runtime_error("Nie udalo sie otworzyc pliku csv: " + grasp_progress_csv_path);
                    }
                }

                std::cout << "========== RUN " << run << " ===========\n\n";

                Solution best = grasp_algorithm(problem, grasp_progress_output, grasp_config, rng);
                results.push_back(best.objective_value);

                if (!has_best_solution || best.objective_value < best_overall.objective_value) {
                    best_overall = best;
                    has_best_solution = true;
                }

                print_solution("Najlepsze rozwiazanie GRASP:", best);
                std::cout << '\n';
            }

            Stats stats = calculate_stats(results);
            print_stats("Podsumowanie GRASP:", stats);
            std::cout << '\n';

            auto end_time = std::chrono::steady_clock::now();
            long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            summary_row(
                SUMMARY_CSV_PATH,
                instance_name,
                "GRASP",
                GRASP_RUNS,
                stats.best,
                stats.worst,
                stats.avg,
                stats.std,
                elapsed_ms
            );
            print_solution("Najlepsze rozwiazanie GRASP ze wszystkich runow:", best_overall);
            return 0;
        }
    }

    return 0;
}

Problem load_problem(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Nie udalo sie otworzyc " + path);
    }

    Problem problem;
    input >> problem.facilities >> problem.customers;

    problem.capacities.resize(problem.facilities);
    problem.opening_costs.resize(problem.facilities);

    for (int facility = 0; facility < problem.facilities; ++facility) {
        input >> problem.capacities[facility] >> problem.opening_costs[facility];
    }

    problem.demands.resize(problem.customers);
    problem.assignment_costs.assign(problem.customers, std::vector<double>(problem.facilities, 0.0));

    for (int customer = 0; customer < problem.customers; ++customer) {
        input >> problem.demands[customer];

        for (int facility = 0; facility < problem.facilities; ++facility) {
            input >> problem.assignment_costs[customer][facility];
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
    solution.facility_customer_counts.assign(problem.facilities, 0);

    for (int customer = 0; customer < static_cast<int>(solution.customer_assignment.size()); ++customer) {
        int facility = solution.customer_assignment[customer];
        if (facility < 0 || facility >= problem.facilities) {
            continue;
        }

        solution.facility_open[facility] = true;
        solution.facility_loads[facility] += problem.demands[customer];
        solution.facility_customer_counts[facility] += 1;
    }
}

void move_customer_inplace(const Problem& problem, Solution& solution, int customer, int new_facility) {
    int old_facility = solution.customer_assignment[customer];
    if (old_facility == new_facility) {
        return;
    }

    int demand = problem.demands[customer];

    if (old_facility >= 0 && old_facility < problem.facilities) {
        solution.facility_loads[old_facility] -= demand;
        solution.facility_customer_counts[old_facility] -= 1;
        solution.facility_open[old_facility] = (solution.facility_customer_counts[old_facility] > 0);
    }

    solution.customer_assignment[customer] = new_facility;
    solution.facility_loads[new_facility] += demand;
    solution.facility_customer_counts[new_facility] += 1;
    solution.facility_open[new_facility] = true;
}

double evaluate_solution(const Problem& problem, Solution& solution) {
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

Stats calculate_stats(const std::vector<double>& values) {
    Stats stats;
    if (values.empty()) {
        return stats;
    }

    stats.best = values[0];
    stats.worst = values[0];

    double sum = 0.0;
    for (double value : values) {
        stats.best = std::min(stats.best, value);
        stats.worst = std::max(stats.worst, value);
        sum += value;
    }

    stats.avg = sum / static_cast<double>(values.size());

    double squared_diff_sum = 0.0;
    for (double value : values) {
        double diff = value - stats.avg;
        squared_diff_sum += diff * diff;
    }

    stats.std = std::sqrt(squared_diff_sum / static_cast<double>(values.size()));
    return stats;
}

std::vector<double> population_scores(const std::vector<Solution>& population) {
    std::vector<double> scores;
    scores.reserve(population.size());

    for (const Solution& solution : population) {
        scores.push_back(solution.objective_value);
    }

    return scores;
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
}

void print_solution(const std::string& name, const Solution& solution) {
    std::cout << name << '\n';
    std::cout << "Otwarte magazyny: ";
    for (int facility = 0; facility < static_cast<int>(solution.facility_open.size()); ++facility) {
        if (solution.facility_open[facility]) {
            std::cout << facility << ' ';
        }
    }
    std::cout << '\n';

    std::cout << "Obciazenia magazynow: ";
    for (int facility = 0; facility < static_cast<int>(solution.facility_loads.size()); ++facility) {
        if (!solution.facility_open[facility]) {
            continue;
        }
        std::cout << "[" << facility << ": " << solution.facility_loads[facility] << "] ";
    }
    std::cout << '\n';

    std::cout << "Przypisania per magazyn:\n";
    for (int facility = 0; facility < static_cast<int>(solution.facility_open.size()); ++facility) {
        if (!solution.facility_open[facility]) {
            continue;
        }

        std::cout << "M" << facility << ": ";

        for (int customer = 0; customer < static_cast<int>(solution.customer_assignment.size()); ++customer) {
            if (solution.customer_assignment[customer] == facility) {
                std::cout << "K" << customer << ' ';
            }
        }
        std::cout << '\n';
    }

    std::cout << "Koszt rozwiazania: " << solution.objective_value << '\n';
}

void print_stats(const std::string& name, const Stats& stats) {
    std::cout << name << '\n';
    std::cout << "best:  " << stats.best << '\n';
    std::cout << "worst: " << stats.worst << '\n';
    std::cout << "avg:   " << stats.avg << '\n';
    std::cout << "std:   " << stats.std << '\n';
}

void summary_row(
    const std::string& csv_path,
    const std::string& instance_name,
    const std::string& method_name,
    int runs,
    double best,
    std::optional<double> worst,
    std::optional<double> avg,
    std::optional<double> std,
    long long time_ms
) {
    bool write_header = false;
    {
        std::ifstream check(csv_path);
        if (!check || check.peek() == std::ifstream::traits_type::eof()) {
            write_header = true;
        }
    }

    std::ofstream output(csv_path, std::ios::app);
    if (!output) {
        throw std::runtime_error("Nie udalo sie otworzyc pliku csv: " + csv_path);
    }

    if (write_header) {
        output << "instance,method,runs,best,worst,avg,std,time_ms\n";
    }

    output << instance_name << ','
           << method_name << ','
           << runs << ','
           << best << ','
           ;

    if (worst.has_value()) {
        output << *worst;
    }
    output << ',';

    if (avg.has_value()) {
        output << std::fixed << std::setprecision(3) << *avg;
    }
    output << ',';

    if (std.has_value()) {
        output << std::fixed << std::setprecision(3) << *std;
    }
    output << ','
           << time_ms << '\n';
}

std::string make_ea_csv_path(const std::string& instance_name, const EAConfig& config, int run_number) {
    std::filesystem::create_directories(EA_RUNS_DIR);

    std::string file_name = instance_name
        + "_pop" + std::to_string(config.pop_size)
        + "_gen" + std::to_string(config.gen)
        + "_px" + std::to_string(static_cast<int>(config.cross_pro * 100))
        + "_pm" + std::to_string(static_cast<int>(config.mutation_pro * 1000))
        + "_tour" + std::to_string(config.tour_size)
        + "_run" + std::to_string(run_number)
        + ".csv";

    return EA_RUNS_DIR + "/" + file_name;
}

void create_ea_csv(const std::string& csv_path) {
    std::ofstream output(csv_path);
    if (!output) {
        throw std::runtime_error("Nie udalo sie utworzyc pliku csv: " + csv_path);
    }

    output << "generation,best,avg,worst,std\n";
}

void ea_progress_row(std::ofstream& csv_output, int generation, const Stats& stats) {
    if (!csv_output.is_open()) {
        return;
    }

    csv_output << generation << ','
               << stats.best << ','
               << std::fixed << std::setprecision(3) << stats.avg << ','
               << stats.worst << ','
               << std::fixed << std::setprecision(3) << stats.std << '\n';
}

std::string make_sa_csv_path(const std::string& instance_name, const SAConfig& config, int run_number) {
    std::filesystem::create_directories(SA_RUNS_DIR);

    std::string file_name = instance_name
        + "_t0" + std::to_string(static_cast<int>(config.initial_temp))
        + "_cool" + std::to_string(static_cast<int>(config.cooling_rate * 1000))
        + "_iter" + std::to_string(config.iter_per_temp)
        + "_run" + std::to_string(run_number)
        + ".csv";

    return SA_RUNS_DIR + "/" + file_name;
}

void create_sa_csv(const std::string& csv_path) {
    std::ofstream output(csv_path);
    if (!output) {
        throw std::runtime_error("Nie udalo sie utworzyc pliku csv: " + csv_path);
    }

    output << "iteration,current,best,temp\n";
}

void sa_progress_row(std::ofstream& csv_output, int iteration, double current_value, double best_value, double temperature) {
    if (!csv_output.is_open()) {
        return;
    }

    csv_output << iteration << ','
               << current_value << ','
               << best_value << ','
               << std::fixed << std::setprecision(6) << temperature << '\n';
}

std::string make_vns_csv_path(const std::string& instance_name, const VNSConfig& config, int run_number) {
    std::filesystem::create_directories(VNS_RUNS_DIR);

    std::string file_name = instance_name
        + "_iter" + std::to_string(config.max_iterations)
        + "_shake" + std::to_string(config.shake_attempts_per_neighborhood)
        + "_ls" + std::to_string(config.local_search_attempts)
        + "_run" + std::to_string(run_number)
        + ".csv";

    return VNS_RUNS_DIR + "/" + file_name;
}

void create_vns_csv(const std::string& csv_path) {
    std::ofstream output(csv_path);
    if (!output) {
        throw std::runtime_error("Nie udalo sie utworzyc pliku csv: " + csv_path);
    }

    output << "iteration,shaken,after_local_search,best\n";
}

void vns_progress_row(std::ofstream& csv_output, int iteration, double shaken_value, double after_local_search_value, double best_value) {
    if (!csv_output.is_open()) {
        return;
    }

    csv_output << iteration << ','
               << shaken_value << ','
               << after_local_search_value << ','
               << best_value << '\n';
}

std::string make_grasp_csv_path(const std::string& instance_name, const GRASPConfig& config, int run_number) {
    std::filesystem::create_directories(GRASP_RUNS_DIR);

    std::string file_name = instance_name
        + "_iter" + std::to_string(config.iterations)
        + "_rcl" + std::to_string(config.rcl_size)
        + "_ls" + std::to_string(config.local_search_attempts)
        + "_run" + std::to_string(run_number)
        + ".csv";

    return GRASP_RUNS_DIR + "/" + file_name;
}

void create_grasp_csv(const std::string& csv_path) {
    std::ofstream output(csv_path);
    if (!output) {
        throw std::runtime_error("Nie udalo sie utworzyc pliku csv: " + csv_path);
    }

    output << "iteration,constructed,improved,best\n";
}

void grasp_progress_row(std::ofstream& csv_output, int iteration, double constructed_value, double improved_value, double best_value) {
    if (!csv_output.is_open()) {
        return;
    }

    csv_output << iteration << ','
               << constructed_value << ','
               << improved_value << ','
               << best_value << '\n';
}

Solution random_solution(const Problem& problem, std::mt19937& rng) {
    const int max_attempts = 100;

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
            std::vector<int> available_facilities;
            available_facilities.reserve(problem.facilities);

            for (int facility = 0; facility < problem.facilities; ++facility) {
                if (remaining_capacity[facility] >= problem.demands[customer]) {
                    available_facilities.push_back(facility);
                }
            }

            if (available_facilities.empty()) {
                success = false;
                break;
            }

            std::uniform_int_distribution<int> distribution(0,static_cast<int>(available_facilities.size()) - 1);
            int chosen_facility = available_facilities[distribution(rng)];

            solution.customer_assignment[customer] = chosen_facility;
            remaining_capacity[chosen_facility] -= problem.demands[customer];
        }

        if (success) {
            update_solution_state(problem, solution);
            solution.objective_value = evaluate_solution(problem, solution);
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

    update_solution_state(problem, solution);
    solution.objective_value = evaluate_solution(problem, solution);
    return solution;
}

std::vector<Solution> initialize_population(const Problem& problem, const EAConfig& config, std::mt19937& rng) {
    std::vector<Solution> population;
    population.reserve(config.pop_size);

    while (static_cast<int>(population.size()) < config.pop_size) {
        population.push_back(random_solution(problem, rng));
    }

    return population;
}

int tournament_selection(const std::vector<Solution>& population, int tournament_size, std::mt19937& rng) {
    std::uniform_int_distribution<int> index_draw(0, static_cast<int>(population.size()) - 1);
    int best_index = index_draw(rng);

    for (int i = 1; i < tournament_size; ++i) {
        int candidate_index = index_draw(rng);
        if (population[candidate_index].objective_value < population[best_index].objective_value) {
            best_index = candidate_index;
        }
    }

    return best_index;
}

Solution evolutionary_algorithm(const Problem& problem, std::ofstream& csv_output, const EAConfig& config, std::mt19937& rng) {
    std::vector<Solution> population = initialize_population(problem, config, rng);
    Solution best = population[0];

    for (const Solution& solution : population) {
        if (solution.objective_value < best.objective_value) {
            best = solution;
        }
    }

    std::uniform_real_distribution<double> probability_draw(0.0, 1.0);

    for (int generation = 0; generation < config.gen; ++generation) {
        Stats current_stats = calculate_stats(population_scores(population));
        ea_progress_row(csv_output, generation + 1, current_stats);
        if (EA_VERBOSE) {
            std::cout << "Gen. " << generation + 1
                      << ", Best: " << current_stats.best
                      << ", Avg: " << current_stats.avg
                      << ", Worst: " << current_stats.worst << '\n';
        }

        std::vector<Solution> next_population;
        next_population.reserve(config.pop_size);

        while (static_cast<int>(next_population.size()) < config.pop_size) {
            int parent1_index = tournament_selection(population, config.tour_size, rng);
            int parent2_index = tournament_selection(population, config.tour_size, rng);

            while (parent2_index == parent1_index) {
                parent2_index = tournament_selection(population, config.tour_size, rng);
            }

            Solution child;
            if (probability_draw(rng) < config.cross_pro) {
                child = crossover(
                    problem,
                    population[parent1_index],
                    population[parent2_index],
                    config.better_parent_bias,
                    rng
                );
            } else {
                child = population[parent1_index];
            }

            mutate_solution(problem, child, rng, config.mutation_pro);
            next_population.push_back(child);

            if (child.objective_value < best.objective_value) {
                best = child;
            }
        }

        population = next_population;
    }

    return best;
}

Solution sa_neighbor(const Problem& problem, const Solution& current, std::mt19937& rng) {
    Solution neighbor = current;

    std::uniform_int_distribution<int> customer_draw(0, problem.customers - 1);
    int customer = customer_draw(rng);
    mutate_customer(problem, neighbor, customer, rng);

    return neighbor;
}

Solution simulated_annealing(const Problem& problem, std::ofstream& csv_output, const SAConfig& config, std::mt19937& rng) {
    Solution current = random_solution(problem, rng);
    Solution best = current;

    std::uniform_real_distribution<double> probability_draw(0.0, 1.0);
    double temperature = config.initial_temp;
    int iteration_number = 1;

    while (temperature > config.min_temp) {
        for (int iteration = 0; iteration < config.iter_per_temp; ++iteration) {
            Solution neighbor = sa_neighbor(problem, current, rng);

            double delta = neighbor.objective_value - current.objective_value;
            if (delta < 0) {
                current = neighbor;
            } else {
                double accept_prob = std::exp(-delta / temperature);
                if (probability_draw(rng) < accept_prob) {
                    current = neighbor;
                }
            }

            if (current.objective_value < best.objective_value) {
                best = current;
            }

            sa_progress_row(
                csv_output,
                iteration_number,
                current.objective_value,
                best.objective_value,
                temperature
            );
            ++iteration_number;
        }

        temperature *= config.cooling_rate;
    }

    return best;
}

bool repair_solution(const Problem& problem, Solution& solution, std::mt19937& rng) {
    update_solution_state(problem, solution);

    const int max_repairs = problem.customers * problem.facilities;

    for (int step = 0; step < max_repairs; ++step) {
        int overloaded_facility = -1;

        for (int facility = 0; facility < problem.facilities; ++facility) {
            if (solution.facility_loads[facility] > problem.capacities[facility]) {
                overloaded_facility = facility;
                break;
            }
        }

        if (overloaded_facility == -1) {
            solution.objective_value = evaluate_solution(problem, solution);
            return true;
        }

        std::vector<int> facility_customers;
        facility_customers.reserve(problem.customers);
        for (int customer = 0; customer < problem.customers; ++customer) {
            if (solution.customer_assignment[customer] == overloaded_facility) {
                facility_customers.push_back(customer);
            }
        }

        if (facility_customers.empty()) {
            return false;
        }

        std::shuffle(facility_customers.begin(), facility_customers.end(), rng);

        bool repaired_step = false;

        for (int customer : facility_customers) {
            int best_new_facility = -1;
            double best_increment = std::numeric_limits<double>::max();

            for (int facility = 0; facility < problem.facilities; ++facility) {
                if (facility == overloaded_facility) {
                    continue;
                }

                if (solution.facility_loads[facility] + problem.demands[customer] <= problem.capacities[facility]) {
                    if (DETERMINISTIC_REPAIR_TARGET) {
                        double increment = problem.assignment_costs[customer][facility];
                        if (!solution.facility_open[facility]) {
                            increment += problem.opening_costs[facility];
                        }

                        if (increment < best_increment) {
                            best_increment = increment;
                            best_new_facility = facility;
                        }
                    } else {
                        best_new_facility = facility;
                        break;
                    }
                }
            }

            if (best_new_facility == -1) {
                continue;
            }

            if (!DETERMINISTIC_REPAIR_TARGET) {
                std::vector<int> candidate_facilities;
                candidate_facilities.reserve(problem.facilities);

                for (int facility = 0; facility < problem.facilities; ++facility) {
                    if (facility == overloaded_facility) {
                        continue;
                    }

                    if (solution.facility_loads[facility] + problem.demands[customer] <= problem.capacities[facility]) {
                        candidate_facilities.push_back(facility);
                    }
                }

                if (candidate_facilities.empty()) {
                    continue;
                }

                std::uniform_int_distribution<int> distribution(0, static_cast<int>(candidate_facilities.size()) - 1);
                best_new_facility = candidate_facilities[distribution(rng)];
            }

            move_customer_inplace(problem, solution, customer, best_new_facility);
            repaired_step = true;
            break;
        }

        if (!repaired_step) {
            return false;
        }
    }

    return false;
}

void mutate_customer(const Problem& problem, Solution& solution, int customer, std::mt19937& rng) {
    int current_facility = solution.customer_assignment[customer];

    std::vector<int> candidate_facilities;
    candidate_facilities.reserve(problem.facilities);
    for (int facility = 0; facility < problem.facilities; ++facility) {
        if (facility != current_facility) {
            candidate_facilities.push_back(facility);
        }
    }

    if (candidate_facilities.empty()) {
        return;
    }

    std::uniform_int_distribution<int> distribution(0, static_cast<int>(candidate_facilities.size()) - 1);
    int new_facility = candidate_facilities[distribution(rng)];

    Solution original_solution = solution;
    move_customer_inplace(problem, solution, customer, new_facility);

    if (!repair_solution(problem, solution, rng)) {
        solution = original_solution;
    }
}

void mutate_solution(const Problem& problem, Solution& solution, std::mt19937& rng, double mutation_probability) {
    std::uniform_real_distribution<double> probability_draw(0.0, 1.0);
    if (probability_draw(rng) < mutation_probability) {
        std::uniform_int_distribution<int> customer_draw(0, problem.customers - 1);
        int customer = customer_draw(rng);
        mutate_customer(problem, solution, customer, rng);
    }
}

Solution crossover(
    const Problem& problem,
    const Solution& parent1,
    const Solution& parent2,
    double better_parent_bias,
    std::mt19937& rng
) {
    const Solution* better_parent = &parent1;
    const Solution* worse_parent = &parent2;

    if (parent2.objective_value < parent1.objective_value) {
        better_parent = &parent2;
        worse_parent = &parent1;
    }

    Solution child;
    child.customer_assignment.assign(problem.customers, -1);

    std::uniform_real_distribution<double> probability_draw(0.0, 1.0);

    for (int customer = 0; customer < problem.customers; ++customer) {
        if (probability_draw(rng) < better_parent_bias) {
            child.customer_assignment[customer] = better_parent->customer_assignment[customer];
        } else {
            child.customer_assignment[customer] = worse_parent->customer_assignment[customer];
        }
    }

    if (!repair_solution(problem, child, rng)) {
        child = *better_parent;
        return child;
    }
    return child;
}

Solution local_search_move_one(
    const Problem& problem,
    const Solution& start,
    int max_attempts,
    int neighborhood,
    int& log_iteration,
    double& best_value,
    std::ofstream& csv_output,
    std::mt19937& rng
) {
    (void)neighborhood;
    (void)log_iteration;
    (void)csv_output;

    Solution best = start;
    std::vector<int> customers(problem.customers);
    std::iota(customers.begin(), customers.end(), 0);
    int attempts_used = 0;

    bool improved = true;
    while (improved && attempts_used < max_attempts) {
        improved = false;
        std::shuffle(customers.begin(), customers.end(), rng);

        for (int customer : customers) {
            if (attempts_used >= max_attempts) {
                break;
            }

            std::vector<int> facilities(problem.facilities);
            std::iota(facilities.begin(), facilities.end(), 0);
            std::shuffle(facilities.begin(), facilities.end(), rng);

            int current_facility = best.customer_assignment[customer];

            for (int facility : facilities) {
                if (facility == current_facility) {
                    continue;
                }

                if (attempts_used >= max_attempts) {
                    break;
                }

                ++attempts_used;
                Solution candidate = best;
                move_customer_inplace(problem, candidate, customer, facility);

                if (!repair_solution(problem, candidate, rng)) {
                    continue;
                }

                if (candidate.objective_value < best.objective_value) {
                    best = candidate;
                    if (best.objective_value < best_value) {
                        best_value = best.objective_value;
                    }
                    improved = true;
                    break;
                }
            }

            if (improved) {
                break;
            }
        }
    }

    return best;
}

bool shake_swap_two(const Problem& problem, Solution& solution, std::mt19937& rng) {
    if (problem.customers < 2) {
        return false;
    }

    std::uniform_int_distribution<int> customer_draw(0, problem.customers - 1);
    const int max_attempts = problem.customers * 2;

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        int first_customer = customer_draw(rng);
        int second_customer = customer_draw(rng);

        if (first_customer == second_customer) {
            continue;
        }

        int first_facility = solution.customer_assignment[first_customer];
        int second_facility = solution.customer_assignment[second_customer];

        if (first_facility == second_facility) {
            continue;
        }

        Solution candidate = solution;
        candidate.customer_assignment[first_customer] = second_facility;
        candidate.customer_assignment[second_customer] = first_facility;

        if (repair_solution(problem, candidate, rng)) {
            solution = candidate;
            return true;
        }
    }

    return false;
}

bool shake_move_two(const Problem& problem, Solution& solution, std::mt19937& rng) {
    if (problem.customers < 2) {
        return false;
    }

    Solution candidate = solution;
    std::vector<int> customers(problem.customers);
    std::iota(customers.begin(), customers.end(), 0);
    std::shuffle(customers.begin(), customers.end(), rng);
    int moved_customers = 0;

    for (int customer : customers) {
        if (moved_customers >= 2) {
            break;
        }

        int original_facility = candidate.customer_assignment[customer];
        mutate_customer(problem, candidate, customer, rng);
        if (candidate.customer_assignment[customer] != original_facility) {
            ++moved_customers;
        }
    }

    if (moved_customers < 2) {
        return false;
    }

    solution = candidate;
    return true;
}

bool shake_partial_facility_reallocation(const Problem& problem, Solution& solution, std::mt19937& rng) {
    update_solution_state(problem, solution);

    std::vector<int> open_facilities;
    open_facilities.reserve(problem.facilities);
    for (int facility = 0; facility < problem.facilities; ++facility) {
        if (solution.facility_open[facility]) {
            open_facilities.push_back(facility);
        }
    }

    if (open_facilities.empty()) {
        return false;
    }

    std::shuffle(open_facilities.begin(), open_facilities.end(), rng);

    for (int source_facility : open_facilities) {
        std::vector<int> facility_customers;
        facility_customers.reserve(problem.customers);
        for (int customer = 0; customer < problem.customers; ++customer) {
            if (solution.customer_assignment[customer] == source_facility) {
                facility_customers.push_back(customer);
            }
        }

        if (facility_customers.size() < 2) {
            continue;
        }

        std::shuffle(facility_customers.begin(), facility_customers.end(), rng);
        int customers_to_move = std::min(3, static_cast<int>(facility_customers.size()));

        Solution candidate = solution;
        int moved_customers = 0;

        for (int index = 0; index < customers_to_move; ++index) {
            int customer = facility_customers[index];
            int original_facility = candidate.customer_assignment[customer];
            mutate_customer(problem, candidate, customer, rng);
            if (candidate.customer_assignment[customer] != original_facility) {
                ++moved_customers;
            }
        }

        if (moved_customers > 0) {
            solution = candidate;
            return true;
        }
    }

    return false;
}

Solution variable_neighborhood_search(const Problem& problem, std::ofstream& csv_output, const VNSConfig& config, std::mt19937& rng) {
    Solution current = random_solution(problem, rng);
    Solution initial_solution = current;
    int log_iteration = 1;
    double best_value = current.objective_value;
    std::ofstream null_csv;
    int dummy_iteration = 0;

    current = local_search_move_one(problem, current, config.local_search_attempts, 0, dummy_iteration, best_value, null_csv, rng);
    Solution best = current;
    best_value = best.objective_value;
    vns_progress_row(csv_output, log_iteration, initial_solution.objective_value, current.objective_value, best_value);
    ++log_iteration;
    const int neighborhood_count = 3;

    for (int iteration = 0; iteration < config.max_iterations; ++iteration) {
        int neighborhood = 1;

        while (neighborhood <= neighborhood_count) {
            int used_neighborhood = neighborhood;
            bool shaken = false;
            Solution shaken_solution = current;

            for (int attempt = 0; attempt < config.shake_attempts_per_neighborhood; ++attempt) {
                shaken_solution = current;

                if (used_neighborhood == 1) {
                    shaken = shake_swap_two(problem, shaken_solution, rng);
                } else if (used_neighborhood == 2) {
                    shaken = shake_move_two(problem, shaken_solution, rng);
                } else {
                    shaken = shake_partial_facility_reallocation(problem, shaken_solution, rng);
                }

                if (shaken) {
                    break;
                }
            }

            if (!shaken) {
                ++neighborhood;
                continue;
            }

            Solution candidate = local_search_move_one(
                problem,
                shaken_solution,
                config.local_search_attempts,
                used_neighborhood,
                dummy_iteration,
                best_value,
                null_csv,
                rng
            );

            if (candidate.objective_value < current.objective_value) {
                current = candidate;
                if (current.objective_value < best.objective_value) {
                    best = current;
                    best_value = best.objective_value;
                }
                vns_progress_row(csv_output, log_iteration, shaken_solution.objective_value, candidate.objective_value, best_value);
                ++log_iteration;
                neighborhood = 1;
            } else {
                vns_progress_row(csv_output, log_iteration, shaken_solution.objective_value, candidate.objective_value, best_value);
                ++log_iteration;
                ++neighborhood;
            }
        }
    }

    return best;
}

Solution grasp_construct_solution(const Problem& problem, const GRASPConfig& config, std::mt19937& rng) {
    const int max_attempts = 100;
    std::vector<int> customer_order(problem.customers);
    std::iota(customer_order.begin(), customer_order.end(), 0);
    std::stable_sort(
        customer_order.begin(),
        customer_order.end(),
        [&problem](int left_customer, int right_customer) {
            return problem.demands[left_customer] > problem.demands[right_customer];
        }
    );

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        Solution solution;
        solution.customer_assignment.assign(problem.customers, -1);
        std::vector<int> remaining_capacity = problem.capacities;
        std::vector<bool> facility_open(problem.facilities, false);
        bool success = true;

        for (int customer : customer_order) {
            std::vector<std::pair<int, double>> candidates;
            candidates.reserve(problem.facilities);

            for (int facility = 0; facility < problem.facilities; ++facility) {
                if (remaining_capacity[facility] < problem.demands[customer]) {
                    continue;
                }

                double incremental_cost = problem.assignment_costs[customer][facility];
                if (!facility_open[facility]) {
                    incremental_cost += problem.opening_costs[facility];
                }
                candidates.push_back({facility, incremental_cost});
            }

            if (candidates.empty()) {
                success = false;
                break;
            }

            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const auto& left, const auto& right) {
                    return left.second < right.second;
                }
            );

            int rcl_size = std::min(config.rcl_size, static_cast<int>(candidates.size()));
            std::uniform_int_distribution<int> rcl_draw(0, rcl_size - 1);
            int chosen_index = rcl_draw(rng);
            int chosen_facility = candidates[chosen_index].first;

            solution.customer_assignment[customer] = chosen_facility;
            remaining_capacity[chosen_facility] -= problem.demands[customer];
            facility_open[chosen_facility] = true;
        }

        if (success) {
            update_solution_state(problem, solution);
            solution.objective_value = evaluate_solution(problem, solution);
            return solution;
        }
    }

    return random_solution(problem, rng);
}

Solution grasp_algorithm(const Problem& problem, std::ofstream& csv_output, const GRASPConfig& config, std::mt19937& rng) {
    Solution best_overall;
    bool has_best = false;
    std::ofstream null_csv;

    for (int iteration = 0; iteration < config.iterations; ++iteration) {
        Solution current = grasp_construct_solution(problem, config, rng);
        double constructed_value = current.objective_value;
        int dummy_iteration = 0;
        double best_value = current.objective_value;
        current = local_search_move_one(
            problem,
            current,
            config.local_search_attempts,
            0,
            dummy_iteration,
            best_value,
            null_csv,
            rng
        );

        if (!has_best || current.objective_value < best_overall.objective_value) {
            best_overall = current;
            has_best = true;
        }

        grasp_progress_row(
            csv_output,
            iteration + 1,
            constructed_value,
            current.objective_value,
            best_overall.objective_value
        );
    }

    return best_overall;
}

// ── Batch experiment mode ─────────────────────────────────────────────────────

void run_batch_experiments() {
    namespace fs = std::filesystem;
    fs::create_directories(PROJECT_DIR / "data");

    const std::string batch_csv = (PROJECT_DIR / "data" / "batch_summary.csv").string();
    {
        std::ofstream f(batch_csv);
        f << "instance,method,config_tag,runs,best,worst,avg,std,time_ms\n";
    }

    auto record = [&](const std::string& inst, const std::string& method,
                      const std::string& tag, int runs, const Stats& st, long long ms) {
        std::ofstream f(batch_csv, std::ios::app);
        f << inst << ',' << method << ',' << tag << ',' << runs << ','
          << st.best << ',' << st.worst << ','
          << std::fixed << std::setprecision(3) << st.avg << ','
          << std::fixed << std::setprecision(3) << st.std << ','
          << ms << '\n';
        std::cout << "  " << method << " [" << tag << "]"
                  << "  best=" << st.best
                  << "  avg=" << std::fixed << std::setprecision(1) << st.avg
                  << "  (" << ms / 1000 << "s)\n";
        std::cout.flush();
    };

    auto elapsed_ms = [](std::chrono::steady_clock::time_point t0) -> long long {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
    };

    std::mt19937 rng(42);

    std::vector<std::pair<std::string, std::string>> instances = {
        {"cap41_ss.txt",  "cap41_ss"},
        {"cap81_ss.txt",  "cap81_ss"},
        {"cap111_ss.txt", "cap111_ss"},
    };

    // ── 1. BASELINE ───────────────────────────────────────────────────────────
    std::cout << "\n=== 1. BASELINE (all algorithms x 3 instances) ===\n";

    for (auto& [inst_file, inst_name] : instances) {
        Problem problem = load_problem((PROJECT_DIR / inst_file).string());
        std::cout << "\nInstance: " << inst_name << "\n";

        // Random
        {
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            res.reserve(RANDOM_RUNS);
            for (int r = 0; r < RANDOM_RUNS; ++r)
                res.push_back(random_solution(problem, rng).objective_value);
            record(inst_name, "random", "default", RANDOM_RUNS, calculate_stats(res), elapsed_ms(t0));
        }

        // Greedy
        {
            auto t0 = std::chrono::steady_clock::now();
            Solution s = greedy_solution(problem);
            Stats st;
            st.best = st.worst = st.avg = st.std = s.objective_value;
            record(inst_name, "greedy", "default", 1, st, elapsed_ms(t0));
        }

        // EA default
        {
            EAConfig cfg;
            EA_RUNS_DIR = (PROJECT_DIR / "data" / ("ea_default_" + inst_name)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= EA_RUNS; ++run) {
                std::string path = make_ea_csv_path(inst_name, cfg, run);
                std::ofstream csv;
                create_ea_csv(path);
                csv.open(path, std::ios::app);
                res.push_back(evolutionary_algorithm(problem, csv, cfg, rng).objective_value);
            }
            record(inst_name, "EA", "default", EA_RUNS, calculate_stats(res), elapsed_ms(t0));
        }

        // SA default
        {
            SAConfig cfg;
            SA_RUNS_DIR = (PROJECT_DIR / "data" / ("sa_default_" + inst_name)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= SA_RUNS; ++run) {
                std::string path = make_sa_csv_path(inst_name, cfg, run);
                std::ofstream csv;
                create_sa_csv(path);
                csv.open(path, std::ios::app);
                res.push_back(simulated_annealing(problem, csv, cfg, rng).objective_value);
            }
            record(inst_name, "SA", "default", SA_RUNS, calculate_stats(res), elapsed_ms(t0));
        }

        // VNS default
        {
            VNSConfig cfg;
            VNS_RUNS_DIR = (PROJECT_DIR / "data" / ("vns_default_" + inst_name)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= VNS_RUNS; ++run) {
                std::string path = make_vns_csv_path(inst_name, cfg, run);
                std::ofstream csv;
                create_vns_csv(path);
                csv.open(path, std::ios::app);
                res.push_back(variable_neighborhood_search(problem, csv, cfg, rng).objective_value);
            }
            record(inst_name, "VNS", "default", VNS_RUNS, calculate_stats(res), elapsed_ms(t0));
        }

        // GRASP default
        {
            GRASPConfig cfg;
            GRASP_RUNS_DIR = (PROJECT_DIR / "data" / ("grasp_default_" + inst_name)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= GRASP_RUNS; ++run) {
                std::string path = make_grasp_csv_path(inst_name, cfg, run);
                std::ofstream csv;
                create_grasp_csv(path);
                csv.open(path, std::ios::app);
                res.push_back(grasp_algorithm(problem, csv, cfg, rng).objective_value);
            }
            record(inst_name, "GRASP", "default", GRASP_RUNS, calculate_stats(res), elapsed_ms(t0));
        }
    }

    // ── 2. EA PARAMETER SWEEP (cap41_ss) ──────────────────────────────────────
    std::cout << "\n=== 2. EA PARAMETER SWEEP (cap41_ss) ===\n";
    {
        Problem problem = load_problem((PROJECT_DIR / "cap41_ss.txt").string());
        const std::string inst = "cap41_ss";

        // {pop_size, gen, tour_size, mutation_pro, cross_pro, better_parent_bias}
        std::vector<std::pair<std::string, EAConfig>> cfgs = {
            {"pop20",  EAConfig{20,  250,  7, 0.90, 0.85, 0.50}},
            {"pop80",  EAConfig{80,  250,  7, 0.90, 0.85, 0.50}},
            {"gen100", EAConfig{40,  100,  7, 0.90, 0.85, 0.50}},
            {"gen500", EAConfig{40,  500,  7, 0.90, 0.85, 0.50}},
            {"pm050",  EAConfig{40,  250,  7, 0.50, 0.85, 0.50}},
            {"pm070",  EAConfig{40,  250,  7, 0.70, 0.85, 0.50}},
            {"tour3",  EAConfig{40,  250,  3, 0.90, 0.85, 0.50}},
            {"tour10", EAConfig{40,  250, 10, 0.90, 0.85, 0.50}},
        };

        for (auto& [tag, cfg] : cfgs) {
            EA_RUNS_DIR = (PROJECT_DIR / "data" / ("ea_" + tag)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= EA_RUNS; ++run) {
                std::string path = make_ea_csv_path(inst, cfg, run);
                std::ofstream csv;
                create_ea_csv(path);
                csv.open(path, std::ios::app);
                res.push_back(evolutionary_algorithm(problem, csv, cfg, rng).objective_value);
            }
            record(inst, "EA", tag, EA_RUNS, calculate_stats(res), elapsed_ms(t0));
        }
    }

    // ── 3. SA PARAMETER SWEEP (cap41_ss) ──────────────────────────────────────
    std::cout << "\n=== 3. SA PARAMETER SWEEP (cap41_ss) ===\n";
    {
        Problem problem = load_problem((PROJECT_DIR / "cap41_ss.txt").string());
        const std::string inst = "cap41_ss";

        // {initial_temp, cooling_rate, min_temp, iter_per_temp}
        std::vector<std::pair<std::string, SAConfig>> cfgs = {
            {"t0_50k",  SAConfig{50000.0,  0.995, 8.0,  5}},
            {"t0_500k", SAConfig{500000.0, 0.995, 8.0,  5}},
            {"cool990", SAConfig{180000.0, 0.990, 8.0,  5}},
            {"cool999", SAConfig{180000.0, 0.999, 8.0,  5}},
            {"iter2",   SAConfig{180000.0, 0.995, 8.0,  2}},
            {"iter10",  SAConfig{180000.0, 0.995, 8.0, 10}},
        };

        for (auto& [tag, cfg] : cfgs) {
            SA_RUNS_DIR = (PROJECT_DIR / "data" / ("sa_" + tag)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= SA_RUNS; ++run) {
                std::string path = make_sa_csv_path(inst, cfg, run);
                std::ofstream csv;
                create_sa_csv(path);
                csv.open(path, std::ios::app);
                res.push_back(simulated_annealing(problem, csv, cfg, rng).objective_value);
            }
            record(inst, "SA", tag, SA_RUNS, calculate_stats(res), elapsed_ms(t0));
        }
    }

    // ── 4. VNS PARAMETER SWEEP (cap41_ss) ─────────────────────────────────────
    std::cout << "\n=== 4. VNS PARAMETER SWEEP (cap41_ss) ===\n";
    {
        Problem problem = load_problem((PROJECT_DIR / "cap41_ss.txt").string());
        const std::string inst = "cap41_ss";

        // {max_iterations, shake_attempts_per_neighborhood, local_search_attempts}
        std::vector<std::pair<std::string, VNSConfig>> cfgs = {
            {"iter20",  VNSConfig{20,  5,  60}},
            {"iter100", VNSConfig{100, 5,  60}},
            {"shake3",  VNSConfig{50,  3,  60}},
            {"shake8",  VNSConfig{50,  8,  60}},
            {"ls30",    VNSConfig{50,  5,  30}},
            {"ls120",   VNSConfig{50,  5, 120}},
        };

        for (auto& [tag, cfg] : cfgs) {
            VNS_RUNS_DIR = (PROJECT_DIR / "data" / ("vns_" + tag)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= VNS_RUNS; ++run) {
                std::string path = make_vns_csv_path(inst, cfg, run);
                std::ofstream csv;
                create_vns_csv(path);
                csv.open(path, std::ios::app);
                res.push_back(variable_neighborhood_search(problem, csv, cfg, rng).objective_value);
            }
            record(inst, "VNS", tag, VNS_RUNS, calculate_stats(res), elapsed_ms(t0));
        }
    }

    // ── 5. GRASP PARAMETER SWEEP (cap41_ss) ───────────────────────────────────
    std::cout << "\n=== 5. GRASP PARAMETER SWEEP (cap41_ss) ===\n";
    {
        Problem problem = load_problem((PROJECT_DIR / "cap41_ss.txt").string());
        const std::string inst = "cap41_ss";

        // {iterations, rcl_size, local_search_attempts}
        std::vector<std::pair<std::string, GRASPConfig>> cfgs = {
            {"iter50",  GRASPConfig{50,  2, 100}},
            {"iter200", GRASPConfig{200, 2, 100}},
            {"rcl1",    GRASPConfig{100, 1, 100}},
            {"rcl4",    GRASPConfig{100, 4, 100}},
            {"ls50",    GRASPConfig{100, 2,  50}},
            {"ls200",   GRASPConfig{100, 2, 200}},
        };

        for (auto& [tag, cfg] : cfgs) {
            GRASP_RUNS_DIR = (PROJECT_DIR / "data" / ("grasp_" + tag)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= GRASP_RUNS; ++run) {
                std::string path = make_grasp_csv_path(inst, cfg, run);
                std::ofstream csv;
                create_grasp_csv(path);
                csv.open(path, std::ios::app);
                res.push_back(grasp_algorithm(problem, csv, cfg, rng).objective_value);
            }
            record(inst, "GRASP", tag, GRASP_RUNS, calculate_stats(res), elapsed_ms(t0));
        }
    }

    std::cout << "\n=== BATCH COMPLETE ===\n";
    std::cout << "Wyniki: " << batch_csv << "\n";
}

// ── Random-search tuning mode ─────────────────────────────────────────────────
// For each algorithm: sample 100 random configs, run TUNE_RUNS times each on
// cap41_ss to rank configs, then run the winner FINAL_RUNS times on all instances.

void run_tuning() {
    namespace fs = std::filesystem;
    const fs::path tune_dir = PROJECT_DIR / "data" / "tuning";
    fs::create_directories(tune_dir);

    const int TUNE_VARIANTS  = 100;
    const int TUNE_RUNS      = 3;
    const int FINAL_RUNS     = 30;
    const std::string tune_inst_file = "cap41_ss.txt";
    const std::string tune_inst_name = "cap41_ss";
    const std::vector<std::string> final_inst_files = {
        "cap41_ss.txt", "cap81_ss.txt", "cap111_ss.txt"
    };

    Problem tune_problem = load_problem((PROJECT_DIR / tune_inst_file).string());
    std::mt19937 rng(std::random_device{}());
    std::mt19937 prng(42);   // separate, fixed-seed RNG for param sampling

    // final_results.csv – one row per (method, instance) for the best config
    const std::string final_csv = (tune_dir / "final_results.csv").string();
    {
        std::ofstream f(final_csv);
        f << "method,instance,runs,best,worst,avg,std,time_ms\n";
    }

    auto write_final = [&](const std::string& method, const std::string& inst,
                           int runs, const Stats& st, long long ms) {
        std::ofstream f(final_csv, std::ios::app);
        f << method << ',' << inst << ',' << runs << ','
          << st.best << ',' << st.worst << ','
          << std::fixed << std::setprecision(3) << st.avg << ','
          << std::fixed << std::setprecision(3) << st.std << ','
          << ms << '\n';
        std::cout << "    [" << inst << "] best=" << st.best
                  << "  avg=" << std::fixed << std::setprecision(1) << st.avg
                  << "  std=" << std::fixed << std::setprecision(1) << st.std << "\n";
    };

    auto elapsed_ms = [](std::chrono::steady_clock::time_point t0) -> long long {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
    };

    // ── 1. EA ─────────────────────────────────────────────────────────────────
    {
        std::cout << "\n=== EA: random search (" << TUNE_VARIANTS
                  << " configs x " << TUNE_RUNS << " runs) ===\n";

        const std::string csv = (tune_dir / "tuning_EA.csv").string();
        { std::ofstream f(csv); f << "config_id,pop_size,gen,tour_size,mutation_pro,cross_pro,bias,best,avg,std,time_ms\n"; }

        std::uniform_int_distribution<int>     pop_d(10,  100);
        std::uniform_int_distribution<int>     gen_d(50,  600);
        std::uniform_int_distribution<int>    tour_d(2,   15);
        std::uniform_real_distribution<double>  mut_d(0.10, 1.00);
        std::uniform_real_distribution<double>  crs_d(0.50, 1.00);
        std::uniform_real_distribution<double> bias_d(0.30, 0.80);

        EAConfig best_cfg;
        double   best_avg = std::numeric_limits<double>::max();

        for (int v = 1; v <= TUNE_VARIANTS; ++v) {
            EAConfig cfg;
            cfg.pop_size           = pop_d(prng);
            cfg.gen                = gen_d(prng);
            cfg.tour_size          = tour_d(prng);
            cfg.mutation_pro       = mut_d(prng);
            cfg.cross_pro          = crs_d(prng);
            cfg.better_parent_bias = bias_d(prng);

            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int r = 0; r < TUNE_RUNS; ++r) {
                std::ofstream null_csv;
                res.push_back(evolutionary_algorithm(tune_problem, null_csv, cfg, rng).objective_value);
            }
            long long ms = elapsed_ms(t0);
            Stats st = calculate_stats(res);

            { std::ofstream f(csv, std::ios::app);
              f << v << ',' << cfg.pop_size << ',' << cfg.gen << ',' << cfg.tour_size << ','
                << std::fixed << std::setprecision(4) << cfg.mutation_pro << ','
                << cfg.cross_pro << ',' << cfg.better_parent_bias << ','
                << st.best << ',' << st.avg << ',' << st.std << ',' << ms << '\n'; }

            if (st.avg < best_avg) { best_avg = st.avg; best_cfg = cfg; }

            std::cout << "[" << std::setw(3) << v << "/" << TUNE_VARIANTS << "]"
                      << " pop=" << std::setw(3) << cfg.pop_size
                      << " gen=" << std::setw(3) << cfg.gen
                      << " tour=" << std::setw(2) << cfg.tour_size
                      << " pm=" << std::fixed << std::setprecision(2) << cfg.mutation_pro
                      << "  avg=" << std::fixed << std::setprecision(0) << st.avg
                      << (st.avg <= best_avg ? "  <-- best" : "") << "\n";
            std::cout.flush();
        }

        std::cout << "\n  Best EA: pop=" << best_cfg.pop_size << " gen=" << best_cfg.gen
                  << " tour=" << best_cfg.tour_size
                  << " pm=" << std::fixed << std::setprecision(3) << best_cfg.mutation_pro
                  << " px=" << best_cfg.cross_pro << " bias=" << best_cfg.better_parent_bias
                  << "  (avg=" << std::fixed << std::setprecision(0) << best_avg << ")\n";
        std::cout << "  Final: " << FINAL_RUNS << " runs x 3 instances...\n";

        for (const std::string& ifn : final_inst_files) {
            Problem p = load_problem((PROJECT_DIR / ifn).string());
            std::string iname = name_from_path((PROJECT_DIR / ifn).string());
            EA_RUNS_DIR = (tune_dir / ("ea_best_" + iname)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= FINAL_RUNS; ++run) {
                std::string path = make_ea_csv_path(iname, best_cfg, run);
                std::ofstream cf; create_ea_csv(path); cf.open(path, std::ios::app);
                res.push_back(evolutionary_algorithm(p, cf, best_cfg, rng).objective_value);
            }
            write_final("EA", iname, FINAL_RUNS, calculate_stats(res), elapsed_ms(t0));
        }
    }

    // ── 2. SA ─────────────────────────────────────────────────────────────────
    {
        std::cout << "\n=== SA: random search (" << TUNE_VARIANTS
                  << " configs x " << TUNE_RUNS << " runs) ===\n";

        const std::string csv = (tune_dir / "tuning_SA.csv").string();
        { std::ofstream f(csv); f << "config_id,initial_temp,cooling_rate,min_temp,iter_per_temp,best,avg,std,time_ms\n"; }

        // Ranges keep total iterations tractable
        std::uniform_real_distribution<double>  t0_d(5000.0, 300000.0);
        std::uniform_real_distribution<double> cool_d(0.9800,  0.9990);
        std::uniform_real_distribution<double> mint_d(0.5,     10.0);
        std::uniform_int_distribution<int>     iter_d(1, 10);

        SAConfig best_cfg;
        double   best_avg = std::numeric_limits<double>::max();

        for (int v = 1; v <= TUNE_VARIANTS; ++v) {
            SAConfig cfg;
            cfg.initial_temp  = t0_d(prng);
            cfg.cooling_rate  = cool_d(prng);
            cfg.min_temp      = mint_d(prng);
            cfg.iter_per_temp = iter_d(prng);

            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int r = 0; r < TUNE_RUNS; ++r) {
                std::ofstream null_csv;
                res.push_back(simulated_annealing(tune_problem, null_csv, cfg, rng).objective_value);
            }
            long long ms = elapsed_ms(t0);
            Stats st = calculate_stats(res);

            { std::ofstream f(csv, std::ios::app);
              f << v << ','
                << std::fixed << std::setprecision(1) << cfg.initial_temp << ','
                << std::fixed << std::setprecision(5) << cfg.cooling_rate << ','
                << std::fixed << std::setprecision(2) << cfg.min_temp << ','
                << cfg.iter_per_temp << ','
                << st.best << ',' << st.avg << ',' << st.std << ',' << ms << '\n'; }

            if (st.avg < best_avg) { best_avg = st.avg; best_cfg = cfg; }

            std::cout << "[" << std::setw(3) << v << "/" << TUNE_VARIANTS << "]"
                      << " T0=" << std::setw(7) << std::fixed << std::setprecision(0) << cfg.initial_temp
                      << " cool=" << std::fixed << std::setprecision(4) << cfg.cooling_rate
                      << " ipt=" << cfg.iter_per_temp
                      << "  avg=" << std::fixed << std::setprecision(0) << st.avg
                      << (st.avg <= best_avg ? "  <-- best" : "") << "\n";
            std::cout.flush();
        }

        std::cout << "\n  Best SA: T0=" << std::fixed << std::setprecision(0) << best_cfg.initial_temp
                  << " cool=" << std::fixed << std::setprecision(5) << best_cfg.cooling_rate
                  << " min=" << best_cfg.min_temp << " ipt=" << best_cfg.iter_per_temp
                  << "  (avg=" << std::fixed << std::setprecision(0) << best_avg << ")\n";
        std::cout << "  Final: " << FINAL_RUNS << " runs x 3 instances...\n";

        for (const std::string& ifn : final_inst_files) {
            Problem p = load_problem((PROJECT_DIR / ifn).string());
            std::string iname = name_from_path((PROJECT_DIR / ifn).string());
            SA_RUNS_DIR = (tune_dir / ("sa_best_" + iname)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= FINAL_RUNS; ++run) {
                std::string path = make_sa_csv_path(iname, best_cfg, run);
                std::ofstream cf; create_sa_csv(path); cf.open(path, std::ios::app);
                res.push_back(simulated_annealing(p, cf, best_cfg, rng).objective_value);
            }
            write_final("SA", iname, FINAL_RUNS, calculate_stats(res), elapsed_ms(t0));
        }
    }

    // ── 3. VNS ────────────────────────────────────────────────────────────────
    {
        std::cout << "\n=== VNS: random search (" << TUNE_VARIANTS
                  << " configs x " << TUNE_RUNS << " runs) ===\n";

        const std::string csv = (tune_dir / "tuning_VNS.csv").string();
        { std::ofstream f(csv); f << "config_id,max_iterations,shake_attempts,local_search_attempts,best,avg,std,time_ms\n"; }

        std::uniform_int_distribution<int>  iter_d(10, 200);
        std::uniform_int_distribution<int> shake_d(1,  15);
        std::uniform_int_distribution<int>    ls_d(10, 200);

        VNSConfig best_cfg;
        double    best_avg = std::numeric_limits<double>::max();

        for (int v = 1; v <= TUNE_VARIANTS; ++v) {
            VNSConfig cfg;
            cfg.max_iterations                 = iter_d(prng);
            cfg.shake_attempts_per_neighborhood = shake_d(prng);
            cfg.local_search_attempts           = ls_d(prng);

            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int r = 0; r < TUNE_RUNS; ++r) {
                std::ofstream null_csv;
                res.push_back(variable_neighborhood_search(tune_problem, null_csv, cfg, rng).objective_value);
            }
            long long ms = elapsed_ms(t0);
            Stats st = calculate_stats(res);

            { std::ofstream f(csv, std::ios::app);
              f << v << ',' << cfg.max_iterations << ','
                << cfg.shake_attempts_per_neighborhood << ',' << cfg.local_search_attempts << ','
                << st.best << ',' << st.avg << ',' << st.std << ',' << ms << '\n'; }

            if (st.avg < best_avg) { best_avg = st.avg; best_cfg = cfg; }

            std::cout << "[" << std::setw(3) << v << "/" << TUNE_VARIANTS << "]"
                      << " iter=" << std::setw(3) << cfg.max_iterations
                      << " shake=" << std::setw(2) << cfg.shake_attempts_per_neighborhood
                      << " ls="    << std::setw(3) << cfg.local_search_attempts
                      << "  avg="  << std::fixed << std::setprecision(0) << st.avg
                      << (st.avg <= best_avg ? "  <-- best" : "") << "\n";
            std::cout.flush();
        }

        std::cout << "\n  Best VNS: iter=" << best_cfg.max_iterations
                  << " shake=" << best_cfg.shake_attempts_per_neighborhood
                  << " ls=" << best_cfg.local_search_attempts
                  << "  (avg=" << std::fixed << std::setprecision(0) << best_avg << ")\n";
        std::cout << "  Final: " << FINAL_RUNS << " runs x 3 instances...\n";

        for (const std::string& ifn : final_inst_files) {
            Problem p = load_problem((PROJECT_DIR / ifn).string());
            std::string iname = name_from_path((PROJECT_DIR / ifn).string());
            VNS_RUNS_DIR = (tune_dir / ("vns_best_" + iname)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= FINAL_RUNS; ++run) {
                std::string path = make_vns_csv_path(iname, best_cfg, run);
                std::ofstream cf; create_vns_csv(path); cf.open(path, std::ios::app);
                res.push_back(variable_neighborhood_search(p, cf, best_cfg, rng).objective_value);
            }
            write_final("VNS", iname, FINAL_RUNS, calculate_stats(res), elapsed_ms(t0));
        }
    }

    // ── 4. GRASP ──────────────────────────────────────────────────────────────
    {
        std::cout << "\n=== GRASP: random search (" << TUNE_VARIANTS
                  << " configs x " << TUNE_RUNS << " runs) ===\n";

        const std::string csv = (tune_dir / "tuning_GRASP.csv").string();
        { std::ofstream f(csv); f << "config_id,iterations,rcl_size,local_search_attempts,best,avg,std,time_ms\n"; }

        std::uniform_int_distribution<int>  giter_d(20,  300);
        std::uniform_int_distribution<int>    rcl_d(1,    8);
        std::uniform_int_distribution<int>    gls_d(20,  300);

        GRASPConfig best_cfg;
        double      best_avg = std::numeric_limits<double>::max();

        for (int v = 1; v <= TUNE_VARIANTS; ++v) {
            GRASPConfig cfg;
            cfg.iterations            = giter_d(prng);
            cfg.rcl_size              = rcl_d(prng);
            cfg.local_search_attempts = gls_d(prng);

            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int r = 0; r < TUNE_RUNS; ++r) {
                std::ofstream null_csv;
                res.push_back(grasp_algorithm(tune_problem, null_csv, cfg, rng).objective_value);
            }
            long long ms = elapsed_ms(t0);
            Stats st = calculate_stats(res);

            { std::ofstream f(csv, std::ios::app);
              f << v << ',' << cfg.iterations << ',' << cfg.rcl_size << ','
                << cfg.local_search_attempts << ','
                << st.best << ',' << st.avg << ',' << st.std << ',' << ms << '\n'; }

            if (st.avg < best_avg) { best_avg = st.avg; best_cfg = cfg; }

            std::cout << "[" << std::setw(3) << v << "/" << TUNE_VARIANTS << "]"
                      << " iter=" << std::setw(3) << cfg.iterations
                      << " rcl=" << cfg.rcl_size
                      << " ls=" << std::setw(3) << cfg.local_search_attempts
                      << "  avg=" << std::fixed << std::setprecision(0) << st.avg
                      << (st.avg <= best_avg ? "  <-- best" : "") << "\n";
            std::cout.flush();
        }

        std::cout << "\n  Best GRASP: iter=" << best_cfg.iterations
                  << " rcl=" << best_cfg.rcl_size
                  << " ls=" << best_cfg.local_search_attempts
                  << "  (avg=" << std::fixed << std::setprecision(0) << best_avg << ")\n";
        std::cout << "  Final: " << FINAL_RUNS << " runs x 3 instances...\n";

        for (const std::string& ifn : final_inst_files) {
            Problem p = load_problem((PROJECT_DIR / ifn).string());
            std::string iname = name_from_path((PROJECT_DIR / ifn).string());
            GRASP_RUNS_DIR = (tune_dir / ("grasp_best_" + iname)).string();
            auto t0 = std::chrono::steady_clock::now();
            std::vector<double> res;
            for (int run = 1; run <= FINAL_RUNS; ++run) {
                std::string path = make_grasp_csv_path(iname, best_cfg, run);
                std::ofstream cf; create_grasp_csv(path); cf.open(path, std::ios::app);
                res.push_back(grasp_algorithm(p, cf, best_cfg, rng).objective_value);
            }
            write_final("GRASP", iname, FINAL_RUNS, calculate_stats(res), elapsed_ms(t0));
        }
    }

    std::cout << "\n=== TUNING COMPLETE ===\n";
    std::cout << "Tuning CSVs:    " << (tune_dir).string() << "/tuning_*.csv\n";
    std::cout << "Final results:  " << final_csv << "\n";
}
