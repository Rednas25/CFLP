// plot_results.cpp  --  generate SVG plots from CFLP batch experiments
// Compile:  g++ -O2 -std=c++17 plot_results.cpp -o plot_results.exe
// Run after:  CFLP.exe --batch

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

const fs::path PROJECT_DIR = fs::path(__FILE__).parent_path();
const fs::path DATA_DIR    = PROJECT_DIR / "data";
const fs::path PLOTS_DIR   = PROJECT_DIR / "plots";

// ── CSV ───────────────────────────────────────────────────────────────────────

using Row = std::map<std::string, std::string>;

double dval(const Row& r, const std::string& k) {
    auto it = r.find(k);
    if (it == r.end() || it->second.empty()) return 0.0;
    try { return std::stod(it->second); } catch (...) { return 0.0; }
}
std::string sval(const Row& r, const std::string& k) {
    auto it = r.find(k);
    return (it == r.end()) ? "" : it->second;
}

std::vector<Row> read_csv(const fs::path& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::string line;
    if (!std::getline(f, line)) return {};
    std::vector<std::string> hdrs;
    { std::istringstream ss(line); std::string h; while (std::getline(ss, h, ',')) hdrs.push_back(h); }
    std::vector<Row> rows;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        Row row; int i = 0;
        std::istringstream ss(line); std::string v;
        while (std::getline(ss, v, ',') && i < (int)hdrs.size()) row[hdrs[i++]] = v;
        rows.push_back(row);
    }
    return rows;
}

// ── SVG builder ───────────────────────────────────────────────────────────────

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
            << "' width='" << f2(std::max(w, 1.0)) << "' height='" << f2(std::max(h, 0.0))
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

    void polyline(const std::vector<std::pair<double,double>>& pts,
                  const std::string& col, double sw = 2) {
        if (pts.empty()) return;
        buf << "<polyline points='";
        for (auto& [px, py] : pts) buf << f2(px) << "," << f2(py) << " ";
        buf << "' fill='none' stroke='" << col << "' stroke-width='" << f2(sw) << "'/>\n";
    }

    void save(const fs::path& p) {
        std::ofstream f(p);
        f << "<?xml version='1.0' encoding='UTF-8'?>\n"
          << "<svg xmlns='http://www.w3.org/2000/svg' width='" << W << "' height='" << H << "'>\n"
          << buf.str() << "</svg>\n";
        std::cout << "  " << p.filename().string() << "\n";
    }
};

// ── colors ────────────────────────────────────────────────────────────────────

const std::vector<std::string> COL = {
    "#e63946","#f4a261","#2a9d8f","#8ecae6","#a8dadc",
    "#457b9d","#264653","#e9c46a","#e76f51","#023047"
};

// ── axis helpers ──────────────────────────────────────────────────────────────

static void axes_and_grid_y(Svg& svg, double ml, double mr, double mt, double mb,
                              int W, int H, double vmax, int nticks = 5) {
    double pw = W - ml - mr, ph = H - mt - mb;
    for (int i = 0; i <= nticks; ++i) {
        double v = vmax * i / nticks;
        double y = mt + ph - (v / vmax) * ph;
        svg.line(ml, y, ml + pw, y, "#e0e0e0", 1, "4,3");
        std::ostringstream s; s << std::fixed << std::setprecision(0) << v;
        svg.text(ml - 6, y + 4, s.str(), "end", 10, "#666");
    }
    svg.line(ml, mt, ml, mt + ph, "#888", 1.5);
    svg.line(ml, mt + ph, ml + pw, mt + ph, "#888", 1.5);
}

// ── bar chart (single group, best + avg per config) ───────────────────────────

struct BarData { std::string label; double best, avg; };

void bar_chart(const fs::path& out, const std::vector<BarData>& data,
               const std::string& title) {
    if (data.empty()) return;
    const int W = 900, H = 500;
    const double ML = 90, MR = 40, MT = 70, MB = 80;
    double pw = W - ML - MR, ph = H - MT - MB;
    Svg svg(W, H);

    double vmax = 0;
    for (auto& d : data) vmax = std::max({vmax, d.best, d.avg});
    vmax *= 1.12;

    axes_and_grid_y(svg, ML, MR, MT, MB, W, H, vmax);

    int n = (int)data.size();
    double gw = pw / n;

    for (int i = 0; i < n; ++i) {
        double cx  = ML + (i + 0.5) * gw;
        double bw  = gw * 0.35;
        std::string col = COL[i % (int)COL.size()];

        // best (solid)
        double bh_best = (data[i].best / vmax) * ph;
        svg.rect(cx - bw - 1, MT + ph - bh_best, bw, bh_best, col);

        // avg (lighter, hatched via opacity)
        if (data[i].avg > 0) {
            double bh_avg = (data[i].avg / vmax) * ph;
            svg.rect(cx + 1, MT + ph - bh_avg, bw, bh_avg, col + "66", col, 1.5);
        }

        svg.text(cx, MT + ph + 18, data[i].label, "middle", 9, "#333");
    }

    svg.text(ML + pw / 2, MT - 30, title, "middle", 14, "#222");
    svg.text(ML - 65, MT + ph / 2, "Objective value", "middle", 11, "#555", -90);

    // legend
    double lx = ML + pw - 150, ly = MT + 12;
    svg.rect(lx, ly,      14, 12, "#888");
    svg.text(lx + 18, ly + 11, "best",  "start", 10, "#555");
    svg.rect(lx, ly + 20, 14, 12, "#88888844", "#888", 1.5);
    svg.text(lx + 18, ly + 31, "avg",   "start", 10, "#555");

    svg.save(out);
}

// ── grouped bar chart (instances x algorithms) ────────────────────────────────

struct GroupBar { std::string group; std::vector<std::pair<std::string,double>> bars; };

void grouped_bar_chart(const fs::path& out, const std::vector<GroupBar>& groups,
                       const std::string& title) {
    if (groups.empty()) return;
    const int W = 1050, H = 520;
    const double ML = 90, MR = 170, MT = 70, MB = 80;
    double pw = W - ML - MR, ph = H - MT - MB;
    Svg svg(W, H);

    double vmax = 0;
    int nsub = 0;
    for (auto& g : groups) {
        nsub = std::max(nsub, (int)g.bars.size());
        for (auto& [l, v] : g.bars) vmax = std::max(vmax, v);
    }
    vmax *= 1.12;

    axes_and_grid_y(svg, ML, MR, MT, MB, W, H, vmax);

    int ng = (int)groups.size();
    double gw = pw / ng;

    for (int gi = 0; gi < ng; ++gi) {
        double gx = ML + gi * gw;
        double cx = gx + gw / 2;
        double total_bw = gw * 0.80;
        double bw = nsub > 0 ? total_bw / nsub : total_bw;
        double bpad = (gw - total_bw) / 2;

        for (int bi = 0; bi < (int)groups[gi].bars.size(); ++bi) {
            double v  = groups[gi].bars[bi].second;
            double bh = (v / vmax) * ph;
            double bx = gx + bpad + bi * bw;
            svg.rect(bx, MT + ph - bh, bw - 2, bh, COL[bi % (int)COL.size()]);
        }
        svg.text(cx, MT + ph + 18, groups[gi].group, "middle", 10, "#333");
    }

    svg.text(ML + pw / 2, MT - 32, title, "middle", 14, "#222");
    svg.text(ML - 65, MT + ph / 2, "Best objective value", "middle", 11, "#555", -90);

    // legend (right)
    double lx = ML + pw + 15, ly = MT + 20;
    std::vector<std::string> sublabels;
    if (!groups.empty()) for (auto& [l, v] : groups[0].bars) sublabels.push_back(l);
    for (int bi = 0; bi < (int)sublabels.size(); ++bi) {
        svg.rect(lx, ly + bi * 22, 16, 14, COL[bi % (int)COL.size()]);
        svg.text(lx + 21, ly + bi * 22 + 12, sublabels[bi], "start", 10, "#444");
    }

    svg.save(out);
}

// ── line chart ────────────────────────────────────────────────────────────────

struct Series { std::string name; std::string color; std::vector<std::pair<double,double>> pts; };

void line_chart(const fs::path& out, const std::vector<Series>& series,
                const std::string& title, const std::string& xlabel, const std::string& ylabel) {
    if (series.empty()) return;
    const int W = 900, H = 480;
    const double ML = 90, MR = 160, MT = 70, MB = 60;
    double pw = W - ML - MR, ph = H - MT - MB;
    Svg svg(W, H);

    double xmax = 0, ymax = 0, ymin = std::numeric_limits<double>::max();
    for (auto& s : series) for (auto& [x, y] : s.pts) {
        xmax = std::max(xmax, x); ymax = std::max(ymax, y); ymin = std::min(ymin, y);
    }
    if (xmax == 0 || ymin > ymax) return;

    double yr = ymax - ymin;
    if (yr < 1e-6) { ymax += 1; ymin -= 1; yr = 2; }
    ymax += yr * 0.08; ymin -= yr * 0.03; yr = ymax - ymin;

    auto px = [&](double x) { return ML + (x / xmax) * pw; };
    auto py = [&](double y) { return MT + ph - ((y - ymin) / yr) * ph; };

    // gridlines x
    for (int i = 0; i <= 5; ++i) {
        double v = xmax * i / 5, x = px(v);
        svg.line(x, MT, x, MT + ph, "#e0e0e0", 1, "4,3");
        std::ostringstream s; s << std::fixed << std::setprecision(0) << v;
        svg.text(x, MT + ph + 16, s.str(), "middle", 10, "#666");
    }
    // gridlines y
    for (int i = 0; i <= 5; ++i) {
        double v = ymin + yr * i / 5, y = py(v);
        svg.line(ML, y, ML + pw, y, "#e0e0e0", 1, "4,3");
        std::ostringstream s; s << std::fixed << std::setprecision(0) << v;
        svg.text(ML - 6, y + 4, s.str(), "end", 10, "#666");
    }

    svg.line(ML, MT, ML, MT + ph, "#888", 1.5);
    svg.line(ML, MT + ph, ML + pw, MT + ph, "#888", 1.5);

    for (auto& s : series) {
        std::vector<std::pair<double,double>> mapped;
        for (auto& [x, y] : s.pts) mapped.push_back({px(x), py(y)});
        svg.polyline(mapped, s.color, 2.0);
    }

    svg.text(ML + pw / 2, MT - 30, title, "middle", 14, "#222");
    svg.text(ML + pw / 2, MT + ph + 42, xlabel, "middle", 11, "#555");
    svg.text(ML - 65, MT + ph / 2, ylabel, "middle", 11, "#555", -90);

    // legend (right)
    double lx = ML + pw + 15, ly = MT + 20;
    for (int i = 0; i < (int)series.size(); ++i) {
        svg.line(lx, ly + i * 22 + 7, lx + 20, ly + i * 22 + 7, series[i].color, 2.5);
        svg.text(lx + 25, ly + i * 22 + 11, series[i].name, "start", 10, "#444");
    }

    svg.save(out);
}

// ── read averaged progress from a directory of run CSVs ───────────────────────

std::vector<std::pair<double,double>> avg_progress(
    const fs::path& dir, const std::string& xcol, const std::string& ycol,
    int max_pts = 0) {
    if (!fs::exists(dir)) return {};

    std::vector<std::vector<std::pair<double,double>>> all_runs;
    for (auto& e : fs::directory_iterator(dir)) {
        if (e.path().extension() != ".csv") continue;
        auto rows = read_csv(e.path());
        std::vector<std::pair<double,double>> run;
        for (auto& r : rows) run.push_back({dval(r, xcol), dval(r, ycol)});
        if (!run.empty()) all_runs.push_back(run);
    }
    if (all_runs.empty()) return {};

    int n = std::numeric_limits<int>::max();
    for (auto& run : all_runs) n = std::min(n, (int)run.size());

    int step = 1;
    if (max_pts > 0 && n > max_pts) step = n / max_pts;

    std::vector<std::pair<double,double>> result;
    for (int i = 0; i < n; i += step) {
        double x = all_runs[0][i].first, y = 0; int cnt = 0;
        for (auto& run : all_runs) if (i < (int)run.size()) { y += run[i].second; ++cnt; }
        if (cnt > 0) result.push_back({x, y / cnt});
    }
    return result;
}

// normalize x to [0, 100]
static std::vector<std::pair<double,double>> norm_x(std::vector<std::pair<double,double>> pts) {
    if (pts.empty()) return pts;
    double xmax = pts.back().first;
    if (xmax < 1e-9) return pts;
    for (auto& [x, y] : pts) x = x / xmax * 100.0;
    return pts;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    fs::create_directories(PLOTS_DIR);
    std::cout << "Generating SVG plots...\n";

    auto batch = read_csv(DATA_DIR / "batch_summary.csv");
    if (batch.empty()) {
        std::cerr << "ERROR: " << (DATA_DIR / "batch_summary.csv")
                  << " not found. Run CFLP.exe --batch first.\n";
        return 1;
    }

    // ── 1. Baseline comparison (grouped bar: instances x algorithms) ──────────
    {
        const std::vector<std::string> methods = {"random","greedy","EA","SA","VNS","GRASP"};
        std::vector<GroupBar> groups;
        for (const std::string& inst : {"cap41_ss","cap81_ss","cap111_ss"}) {
            GroupBar g; g.group = inst;
            for (auto& meth : methods) {
                for (auto& r : batch) {
                    if (sval(r,"instance")==inst && sval(r,"method")==meth
                        && sval(r,"config_tag")=="default") {
                        g.bars.push_back({meth, dval(r,"best")});
                        break;
                    }
                }
            }
            if (!g.bars.empty()) groups.push_back(g);
        }
        grouped_bar_chart(PLOTS_DIR/"01_baseline_comparison.svg", groups,
                          "Baseline: best objective value (all algorithms x 3 instances)");
    }

    // ── 2-5. Convergence plots (cap41_ss, avg over 10 runs) ──────────────────
    {
        auto pts = avg_progress(DATA_DIR/"ea_default_cap41_ss", "generation", "best", 250);
        if (!pts.empty())
            line_chart(PLOTS_DIR/"02_ea_convergence.svg",
                       {{"EA best (avg 10 runs)", COL[0], pts}},
                       "EA Convergence - cap41_ss", "Generation", "Best objective value");
    }
    {
        auto pts = avg_progress(DATA_DIR/"sa_default_cap41_ss", "iteration", "best", 600);
        if (!pts.empty())
            line_chart(PLOTS_DIR/"03_sa_convergence.svg",
                       {{"SA best (avg 10 runs)", COL[1], pts}},
                       "SA Convergence - cap41_ss", "Iteration", "Best objective value");
    }
    {
        auto pts  = avg_progress(DATA_DIR/"vns_default_cap41_ss",  "iteration", "best",  200);
        auto ptsa = avg_progress(DATA_DIR/"vns_default_cap41_ss",  "iteration", "after_local_search", 200);
        std::vector<Series> s;
        if (!pts.empty())  s.push_back({"Best",              COL[2], pts});
        if (!ptsa.empty()) s.push_back({"After local search", COL[7], ptsa});
        if (!s.empty())
            line_chart(PLOTS_DIR/"04_vns_convergence.svg", s,
                       "VNS Convergence - cap41_ss", "Log iteration", "Objective value");
    }
    {
        auto ptsc = avg_progress(DATA_DIR/"grasp_default_cap41_ss", "iteration", "constructed", 100);
        auto ptsi = avg_progress(DATA_DIR/"grasp_default_cap41_ss", "iteration", "improved",    100);
        auto ptsb = avg_progress(DATA_DIR/"grasp_default_cap41_ss", "iteration", "best",        100);
        std::vector<Series> s;
        if (!ptsc.empty()) s.push_back({"Constructed", COL[4], ptsc});
        if (!ptsi.empty()) s.push_back({"After LS",    COL[3], ptsi});
        if (!ptsb.empty()) s.push_back({"Best",        COL[0], ptsb});
        if (!s.empty())
            line_chart(PLOTS_DIR/"05_grasp_convergence.svg", s,
                       "GRASP Convergence - cap41_ss", "Iteration", "Objective value");
    }

    // ── 6-9. Parameter sweep bar charts ───────────────────────────────────────
    auto sweep = [&](const std::string& method, const std::string& fname, const std::string& title) {
        std::vector<BarData> data;
        for (auto& r : batch) {
            if (sval(r,"instance")=="cap41_ss" && sval(r,"method")==method
                && sval(r,"config_tag")!="default")
                data.push_back({sval(r,"config_tag"), dval(r,"best"), dval(r,"avg")});
        }
        // also add default for reference
        for (auto& r : batch) {
            if (sval(r,"instance")=="cap41_ss" && sval(r,"method")==method
                && sval(r,"config_tag")=="default")
                data.insert(data.begin(), {"default", dval(r,"best"), dval(r,"avg")});
        }
        if (!data.empty()) bar_chart(PLOTS_DIR / fname, data, title);
    };
    sweep("EA",    "06_ea_param_sweep.svg",    "EA Parameter Sweep - cap41_ss");
    sweep("SA",    "07_sa_param_sweep.svg",    "SA Parameter Sweep - cap41_ss");
    sweep("VNS",   "08_vns_param_sweep.svg",   "VNS Parameter Sweep - cap41_ss");
    sweep("GRASP", "09_grasp_param_sweep.svg", "GRASP Parameter Sweep - cap41_ss");

    // ── 10. All algorithms convergence (normalized x) ─────────────────────────
    {
        std::vector<Series> s;
        auto ea    = avg_progress(DATA_DIR/"ea_default_cap41_ss",    "generation", "best", 100);
        auto sa    = avg_progress(DATA_DIR/"sa_default_cap41_ss",    "iteration",  "best", 200);
        auto vns   = avg_progress(DATA_DIR/"vns_default_cap41_ss",   "iteration",  "best", 100);
        auto grasp = avg_progress(DATA_DIR/"grasp_default_cap41_ss", "iteration",  "best", 100);
        if (!ea.empty())    s.push_back({"EA",    COL[0], norm_x(ea)});
        if (!sa.empty())    s.push_back({"SA",    COL[1], norm_x(sa)});
        if (!vns.empty())   s.push_back({"VNS",   COL[2], norm_x(vns)});
        if (!grasp.empty()) s.push_back({"GRASP", COL[3], norm_x(grasp)});
        if (!s.empty())
            line_chart(PLOTS_DIR/"10_convergence_all_algorithms.svg", s,
                       "Algorithm convergence comparison - cap41_ss (normalized)",
                       "Progress [%]", "Best objective value");
    }

    // ── 11. Scalability: best values per algorithm across instances ───────────
    {
        const std::vector<std::string> methods = {"EA","SA","VNS","GRASP"};
        const std::vector<std::string> instances = {"cap41_ss","cap81_ss","cap111_ss"};
        std::vector<Series> s;
        for (int mi = 0; mi < (int)methods.size(); ++mi) {
            std::vector<std::pair<double,double>> pts;
            for (int ii = 0; ii < (int)instances.size(); ++ii) {
                for (auto& r : batch) {
                    if (sval(r,"method")==methods[mi] && sval(r,"instance")==instances[ii]
                        && sval(r,"config_tag")=="default") {
                        pts.push_back({(double)(ii + 1), dval(r,"best")});
                        break;
                    }
                }
            }
            if (!pts.empty()) s.push_back({methods[mi], COL[mi], pts});
        }
        if (!s.empty())
            line_chart(PLOTS_DIR/"11_scalability.svg", s,
                       "Best result per algorithm vs instance size",
                       "Instance (1=cap41, 2=cap81, 3=cap111)", "Best objective value");
    }

    // ── 12-15. Tuning: ranked configs per algorithm ───────────────────────────
    // Shows all 100 configs sorted by avg, best highlighted
    auto plot_ranked = [&](const std::string& method, const fs::path& csv_path) {
        auto rows = read_csv(csv_path);
        if (rows.empty()) return;

        std::vector<std::pair<double,int>> ranked; // (avg, config_id)
        for (auto& r : rows) ranked.push_back({dval(r,"avg"), (int)dval(r,"config_id")});
        std::sort(ranked.begin(), ranked.end());

        const int W = 1000, H = 460;
        const double ML = 90, MR = 40, MT = 70, MB = 70;
        double pw = W - ML - MR, ph = H - MT - MB;
        Svg svg(W, H);

        double vmax = ranked.back().first * 1.05;
        double vmin = ranked.front().first * 0.98;
        double vr   = vmax - vmin;

        // gridlines
        for (int i = 0; i <= 5; ++i) {
            double v = vmin + vr * i / 5;
            double y = MT + ph - ((v - vmin) / vr) * ph;
            svg.line(ML, y, ML + pw, y, "#e0e0e0", 1, "4,3");
            std::ostringstream s; s << std::fixed << std::setprecision(0) << v;
            svg.text(ML - 6, y + 4, s.str(), "end", 10, "#666");
        }
        svg.line(ML, MT, ML, MT + ph, "#888", 1.5);
        svg.line(ML, MT + ph, ML + pw, MT + ph, "#888", 1.5);

        int n = (int)ranked.size();
        double bw = pw / n;
        for (int i = 0; i < n; ++i) {
            double v  = ranked[i].first;
            double bh = ((v - vmin) / vr) * ph;
            double bx = ML + i * bw;
            // best = index 0 (lowest avg) → highlight in red, rest grey
            std::string col = (i == 0) ? "#e63946" : "#b0c4de";
            svg.rect(bx, MT + ph - bh, bw - 1, bh, col);
        }

        svg.text(ML + pw / 2, MT - 30, method + " – 100 configs ranked by avg (cap41_ss)",
                 "middle", 14, "#222");
        svg.text(ML + pw / 2, MT + ph + 40, "Rank (best → worst)", "middle", 11, "#555");
        svg.text(ML - 65, MT + ph / 2, "Avg objective value", "middle", 11, "#555", -90);

        // annotate best
        double best_v = ranked[0].first;
        double best_y = MT + ph - ((best_v - vmin) / vr) * ph;
        std::ostringstream ann; ann << std::fixed << std::setprecision(0) << best_v;
        svg.text(ML + bw / 2, best_y - 6, ann.str(), "middle", 10, "#e63946");

        std::string fname = "tuning_" + method + "_ranked.svg";
        for (char& c : fname) if (c == ' ') c = '_';
        svg.save(PLOTS_DIR / fname);
    };

    {
        fs::path td = DATA_DIR / "tuning";
        plot_ranked("EA",    td / "tuning_EA.csv");
        plot_ranked("SA",    td / "tuning_SA.csv");
        plot_ranked("VNS",   td / "tuning_VNS.csv");
        plot_ranked("GRASP", td / "tuning_GRASP.csv");
    }

    // ── 16. Tuning: final best-config results vs default on all instances ─────
    {
        auto final_rows = read_csv(DATA_DIR / "tuning" / "final_results.csv");
        auto batch_rows = read_csv(DATA_DIR / "batch_summary.csv");
        if (!final_rows.empty() && !batch_rows.empty()) {
            const std::vector<std::string> methods   = {"EA","SA","VNS","GRASP"};
            const std::vector<std::string> instances = {"cap41_ss","cap81_ss","cap111_ss"};

            // GroupBar per instance, bar per method: default vs tuned side by side
            // We do 2 bars per method: default (batch) and tuned (final)
            // Actually: group = instance, bars = (default_EA, tuned_EA, default_SA, tuned_SA, ...)
            // That's too many bars. Better: one plot per instance? Or just tuned best per algorithm.

            // Simpler: grouped bar chart – group=instance, bars=tuned algorithms
            std::vector<GroupBar> groups;
            for (auto& inst : instances) {
                GroupBar g; g.group = inst;
                for (auto& meth : methods) {
                    // tuned value
                    double tuned = 0;
                    for (auto& r : final_rows)
                        if (sval(r,"method")==meth && sval(r,"instance")==inst) {
                            tuned = dval(r,"best"); break; }
                    // default value
                    double def = 0;
                    for (auto& r : batch_rows)
                        if (sval(r,"method")==meth && sval(r,"instance")==inst
                            && sval(r,"config_tag")=="default") {
                            def = dval(r,"best"); break; }
                    if (tuned > 0) g.bars.push_back({meth + "(def=" + std::to_string((int)def) + ")", tuned});
                }
                if (!g.bars.empty()) groups.push_back(g);
            }
            if (!groups.empty())
                grouped_bar_chart(PLOTS_DIR/"tuning_final_comparison.svg", groups,
                                  "Tuned best configs – best value per algorithm x instance");
        }
    }

    std::cout << "Done. Plots saved to: " << PLOTS_DIR << "\n";
    return 0;
}
