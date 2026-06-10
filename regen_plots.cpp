// regen_plots.cpp — budget bar charts matching plots/*_param_sweep.svg style exactly
// Compile: g++ -O2 -std=c++17 -static -static-libgcc -static-libstdc++ regen_plots.cpp -o regen_plots.exe

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;
const fs::path PROJECT_DIR = fs::path(__FILE__).parent_path();
const fs::path DATA2  = PROJECT_DIR / "data2";
const fs::path PLOTS  = DATA2 / "plots";

// ── Svg — identical to plot_results.cpp ──────────────────────────────────────

class Svg {
    int W, H;
    std::ostringstream buf;
    std::string f2(double v) {
        std::ostringstream s; s << std::fixed << std::setprecision(2) << v; return s.str();
    }
public:
    Svg(int w, int h) : W(w), H(h) {
        buf << "<rect width='100%' height='100%' fill='white'/>\n";
    }
    void rect(double x, double y, double w, double h,
              const std::string& fill, const std::string& stk = "none", double sw = 1) {
        buf << "<rect x='" << f2(x) << "' y='" << f2(y)
            << "' width='" << f2(std::max(w,0.5)) << "' height='" << f2(std::max(h,0.0))
            << "' fill='" << fill << "'";
        if (stk != "none") buf << " stroke='" << stk << "' stroke-width='" << f2(sw) << "'";
        buf << " rx='3'/>\n";
    }
    void line(double x1, double y1, double x2, double y2,
              const std::string& col, double sw = 1, const std::string& dash = "") {
        buf << "<line x1='" << f2(x1) << "' y1='" << f2(y1)
            << "' x2='" << f2(x2) << "' y2='" << f2(y2)
            << "' stroke='" << col << "' stroke-width='" << f2(sw) << "'";
        if (!dash.empty()) buf << " stroke-dasharray='" << dash << "'";
        buf << "/>\n";
    }
    void text(double x, double y, const std::string& s,
              const std::string& anchor = "middle", int fs = 12,
              const std::string& col = "#333", double rot = 0) {
        buf << "<text x='" << f2(x) << "' y='" << f2(y)
            << "' text-anchor='" << anchor << "' font-size='" << fs
            << "' fill='" << col << "' font-family='Arial,sans-serif'";
        if (rot != 0) buf << " transform='rotate(" << f2(rot) << "," << f2(x) << "," << f2(y) << ")'";
        buf << ">" << s << "</text>\n";
    }
    void save(const fs::path& p) {
        std::ofstream f(p);
        f << "<?xml version='1.0' encoding='UTF-8'?>\n"
          << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W << "' height='" << H << "'>\n"
          << buf.str() << "</svg>\n";
        std::cout << "  " << p.filename().string() << "\n";
    }
};

// ── colour palette — same as plot_results.cpp ─────────────────────────────────

const std::vector<std::string> COL = {
    "#e63946","#f4a261","#2a9d8f","#8ecae6","#a8dadc",
    "#457b9d","#264653","#e9c46a","#e76f51","#023047"
};

// ── CSV reader ────────────────────────────────────────────────────────────────

struct Row { std::vector<std::string> cols; };
std::vector<Row> read_rows(const fs::path& p) {
    std::ifstream f(p); std::string hdr; std::getline(f, hdr);
    std::vector<Row> rows; std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        Row r; std::istringstream ss(line); std::string tok;
        while (std::getline(ss, tok, ',')) r.cols.push_back(tok);
        rows.push_back(r);
    }
    return rows;
}
double dcol(const Row& r, int i) { try { return std::stod(r.cols.at(i)); } catch(...) { return 0; } }

// ── format a parameter value as a short label ─────────────────────────────────

std::string fmt_param(double v) {
    std::ostringstream s;
    if      (v >= 100000) { s << std::fixed << std::setprecision(0) << v/1000 << "k"; }
    else if (v >= 1000)   { s << std::fixed << std::setprecision(0) << v; }
    else if (v >= 10)     { s << std::fixed << std::setprecision(0) << v; }
    else if (v >= 1)      { s << std::fixed << std::setprecision(1) << v; }
    else                  { s << std::fixed << std::setprecision(3) << v; }
    return s.str();
}

// ── grouped bar chart — identical style to plots/*_param_sweep.svg ────────────
// Configs sorted by best value, top_n shown. Each group: solid bar (best) +
// transparent+border bar (avg). Legend uses rect icons like original.

void budget_bar_chart(const fs::path& out,
                      const std::string& title,
                      const std::string& param_name,
                      std::vector<std::tuple<double,double,double>> configs, // {param_val, best, avg}
                      const std::string& bar_col,
                      int top_n = 30)
{
    if (configs.empty()) return;

    // sort by best (ascending)
    std::sort(configs.begin(), configs.end(), [](const auto& a, const auto& b){
        return std::get<1>(a) < std::get<1>(b);
    });
    if ((int)configs.size() > top_n) configs.resize(top_n);
    int N = (int)configs.size();

    // exact same dimensions as param_sweep charts
    const int    W  = 900, H  = 500;
    const double ML = 90,  MR = 50,  MT = 70, MB = 80;
    const double pw = W - ML - MR;   // 760
    const double ph = H - MT - MB;   // 350
    const double x0 = ML, x1 = ML + pw, y0 = MT, y1 = MT + ph;

    // Y from 0 to max*1.05 (like original, bars start from bottom)
    double ymax = 0;
    for (auto& [p,b,a] : configs) ymax = std::max({ymax, b, a});
    ymax *= 1.05;
    if (ymax < 1) ymax = 1;

    auto py = [&](double v) { return y0 + ph * (1.0 - v / ymax); };

    // bar geometry
    double slot_w  = pw / N;
    double bar_w   = slot_w * 0.40;
    double pair_gap = slot_w * 0.04;

    Svg svg(W, H);

    // horizontal grid only (like original param_sweep)
    for (int i = 0; i <= 5; ++i) {
        double v = ymax * i / 5;
        double yp = py(v);
        svg.line(x0, yp, x1, yp, "#e0e0e0", 1, "4,3");
        std::ostringstream s; s << std::fixed << std::setprecision(0) << v;
        svg.text(x0 - 6, yp + 4, s.str(), "end", 10, "#666");
    }

    // axes
    svg.line(x0, y0, x0, y1, "#888", 1.5);
    svg.line(x0, y1, x1, y1, "#888", 1.5);

    // bars
    for (int i = 0; i < N; ++i) {
        auto& [pv, bv, av] = configs[i];
        double cx  = x0 + (i + 0.5) * slot_w;
        double bx  = cx - pair_gap/2 - bar_w;  // best bar (left)
        double ax  = cx + pair_gap/2;           // avg bar (right)

        // best — solid
        svg.rect(bx, py(bv), bar_w, y1 - py(bv), bar_col);
        // avg — transparent fill + border (like #e6394666 in original)
        svg.rect(ax, py(av), bar_w, y1 - py(av), bar_col + "66", bar_col, 1.5);

        // x label: param value, rotated -45°, centered on group
        svg.text(cx, y1 + 10, fmt_param(pv), "end", 8, "#555", -45);
    }

    // title and axis labels — same positions as original
    svg.text(x0 + pw/2, MT - 30, title, "middle", 14, "#222");
    svg.text(25,         y0 + ph/2, "Objective value", "middle", 11, "#555", -90);
    svg.text(x0 + pw/2,  H - 5,
             param_name + "  (top " + std::to_string(N) + " configs, sorted by best)",
             "middle", 10, "#555");

    // legend — solid rect + transparent rect, like original (at right of chart)
    svg.rect(x1 - 80, y0 + 12, 14, 12, "#888");
    svg.text(x1 - 63,  y0 + 23, "best", "start", 10, "#555");
    svg.rect(x1 - 80, y0 + 32, 14, 12, "#88888844", "#888", 1.5);
    svg.text(x1 - 63,  y0 + 43, "avg",  "start", 10, "#555");

    svg.save(out);
}

// ── generate bar charts for one CSV ───────────────────────────────────────────

void gen_charts(const fs::path& csv,
                const std::string& algo,
                const std::vector<std::pair<int,std::string>>& param_cols,
                int avg_col, int best_col,
                const std::string& bar_color)
{
    auto rows = read_rows(csv);
    if (rows.empty()) { std::cout << "  (no data: " << csv.filename() << ")\n"; return; }

    std::string A = algo; for (auto& c : A) c = std::toupper(c);

    for (auto& [ci, name] : param_cols) {
        std::vector<std::tuple<double,double,double>> configs;
        for (auto& r : rows)
            configs.emplace_back(dcol(r,ci), dcol(r,best_col), dcol(r,avg_col));

        budget_bar_chart(
            PLOTS / (algo + "_" + name + "_top30.svg"),
            A + " Parameter Budget Sweep - cap41_ss",
            name, configs, bar_color);
    }
}

int main() {
    fs::create_directories(PLOTS);

    // EA: 0=config_id,1=pop,2=gen_actual,3=tour,4=pm,5=px,6=bias,7=eval_budget,
    //     8=runs,9=best,10=avg,11=std,12=time_ms
    std::cout << "EA:\n";
    gen_charts(DATA2/"ea_budget.csv", "ea",
        {{1,"pop"},{2,"gen_actual"},{3,"tour"},{4,"pm"},{5,"px"}},
        10, 9, COL[0]);

    // SA: 0=config_id,1=T0,2=cool,3=min_T,4=ipt,5=eval_budget,
    //     6=runs,7=best,8=avg,9=std,10=time_ms
    std::cout << "SA:\n";
    gen_charts(DATA2/"sa_budget.csv", "sa",
        {{1,"T0"},{2,"cool"},{4,"ipt"}},
        8, 7, COL[1]);

    // VNS: 0=config_id,1=iter,2=shake,3=ls_actual,4=eval_budget_approx,
    //      5=runs,6=best,7=avg,8=std,9=time_ms
    std::cout << "VNS:\n";
    gen_charts(DATA2/"vns_budget.csv", "vns",
        {{1,"iter"},{2,"shake"},{3,"ls_actual"}},
        7, 6, COL[2]);

    // GRASP: 0=config_id,1=iter,2=rcl,3=ls_actual,4=eval_budget_approx,
    //        5=runs,6=best,7=avg,8=std,9=time_ms
    std::cout << "GRASP:\n";
    gen_charts(DATA2/"grasp_budget.csv", "grasp",
        {{1,"iter"},{2,"rcl"},{3,"ls_actual"}},
        7, 6, COL[4]);

    std::cout << "Done. " << PLOTS.string() << "\n";
}
