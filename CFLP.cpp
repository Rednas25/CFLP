#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
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
const int RANDOM_RUNS = 10000;
const int EA_RUNS = 10;
const std::string SUMMARY_CSV_PATH = "summary.csv";
const std::string EA_RUNS_DIR = "ea_runs3";
const bool SAVE_PROGRESS_CSV = true;

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

struct EAConfig {
    int pop_size = 40;
    int gen = 250;
    int tour_size = 7;
    double mutation_pro = 0.03;
    double cross_pro = 0.85;
    double better_parent_bias = 0.50;
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
bool is_valid_solution(const Problem& problem, const Solution& solution);
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
    const Stats& stats
);
void summary_row_best_only(
    const std::string& csv_path,
    const std::string& instance_name,
    const std::string& method_name,
    int runs,
    double best
);
std::string make_ea_csv_path(const std::string& instance_name, const EAConfig& config, int run_number);
void create_ea_csv(const std::string& csv_path);
void ea_progress_row(std::ofstream& csv_output, int generation, const Stats& stats);
Solution random_solution(const Problem& problem, std::mt19937& rng);
Solution greedy_solution(const Problem& problem);
std::vector<Solution> initialize_population(const Problem& problem, const EAConfig& config, std::mt19937& rng);
int tournament_selection(const std::vector<Solution>& population, int tournament_size, std::mt19937& rng);
Solution evolutionary_algorithm(const Problem& problem, std::ofstream& csv_output, const EAConfig& config, std::mt19937& rng);
bool repair_solution(const Problem& problem, Solution& solution, std::mt19937& rng);
void mutate_customer(const Problem& problem, Solution& solution, int customer, std::mt19937& rng);
void mutate_solution(const Problem& problem, Solution& solution, std::mt19937& rng, double mutation_probability);
Solution crossover(
    const Problem& problem,
    const Solution& parent1,
    const Solution& parent2,
    double better_parent_bias,
    std::mt19937& rng
);

int main(int argc, char* argv[]) {
    std::string path = DEFAULT_INSTANCE_NAME;
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
    std::cout << "wybor: ";

    int method = 0;
    std::cin >> method;
    std::cout << '\n';

    switch (method) {
        case 1: {
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
            summary_row(SUMMARY_CSV_PATH, instance_name, "random", RANDOM_RUNS, stats);
            return 0;
        }

        case 2: {
            Solution greedy = greedy_solution(problem);
            print_solution("Rozwiazanie zachlanne:", greedy);
            summary_row_best_only(SUMMARY_CSV_PATH, instance_name, "greedy", 1, greedy.objective_value);
            return 0;
        }

        case 3: {
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

            summary_row(SUMMARY_CSV_PATH, instance_name, "EA", EA_RUNS, stats);
            print_solution("Najlepszy osobnik EA ze wszystkich runow:", best_overall);
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
    const Stats& stats
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
        output << "instance,method,runs,best,worst,avg,std\n";
    }

    output << instance_name << ','
           << method_name << ','
           << runs << ','
           << stats.best << ','
           << stats.worst << ','
           << std::fixed << std::setprecision(3) << stats.avg << ','
           << std::fixed << std::setprecision(3) << stats.std << '\n';
}

void summary_row_best_only(
    const std::string& csv_path,
    const std::string& instance_name,
    const std::string& method_name,
    int runs,
    double best
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
        output << "instance,method,runs,best,worst,avg,std\n";
    }

    output << instance_name << ','
           << method_name << ','
           << runs << ','
           << best << ','
           << ','
           << ','
           << '\n';
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
        std::cout << "Gen. " << generation + 1
                  << ", Best: " << current_stats.best
                  << ", Avg: " << current_stats.avg
                  << ", Worst: " << current_stats.worst << '\n';

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
            child.objective_value = evaluate_solution(problem, child);
            next_population.push_back(child);

            if (child.objective_value < best.objective_value) {
                best = child;
            }
        }

        population = next_population;
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
            std::vector<int> candidate_facilities;

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
            int new_facility = candidate_facilities[distribution(rng)];
            solution.customer_assignment[customer] = new_facility;
            update_solution_state(problem, solution);
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
    solution.customer_assignment[customer] = new_facility;

    if (!repair_solution(problem, solution, rng)) {
        solution = original_solution;
    }
}

void mutate_solution(const Problem& problem, Solution& solution, std::mt19937& rng, double mutation_probability) {
    std::uniform_real_distribution<double> probability_draw(0.0, 1.0);

    for (int customer = 0; customer < problem.customers; ++customer) {
        if (probability_draw(rng) < mutation_probability) {
            mutate_customer(problem, solution, customer, rng);
        }
    }

    solution.objective_value = evaluate_solution(problem, solution);
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

    child.objective_value = evaluate_solution(problem, child);
    return child;
}
