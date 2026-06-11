// regen_plots.cpp — plots for data2/ budget experiments
// Ranked bar charts: 100 configs sorted by avg (best highlighted).
// Per-parameter line charts: sorted by param value, avg + best lines.
// Style matches plots/ folder exactly.
// Compile: g++ -O2 -std=c++17 -static -static-libgcc -static-libstdc++ regen_plots.cpp -o regen_plots.exe

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
const fs::path DATA2  = "data2";
const fs::path PLOTS  = fs::path("data2") / "plots";

// ── Svg ───────────────────────────────────────────────────────────────────────

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
        for (auto& [x,y] : pts) buf << f2(x) << "," << f2(y) << " ";
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

// ── CSV ───────────────────────────────────────────────────────────────────────

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

// ── ranked bar chart (style: tuning_*_ranked.svg) ─────────────────────────────
// Bars sorted ascending by avg. Rank-1 bar (best avg) highlighted red.

void ranked_chart(const fs::path& out,
                  const std::string& algo,
                  const std::vector<Row>& rows,
                  int avg_col, int best_col)
{
    if (rows.empty()) return;
    std::string A = algo; for (auto& c : A) c = std::toupper(c);

    // build (avg, best) and sort ascending by avg
    std::vector<std::pair<double,double>> ranked;
    for (auto& r : rows) ranked.push_back({dcol(r, avg_col), dcol(r, best_col)});
    std::sort(ranked.begin(), ranked.end());

    const int    W  = 1000, H  = 460;
    const double ML = 90, MR = 40, MT = 70, MB = 70;
    const double pw = W - ML - MR, ph = H - MT - MB;

    double vmax = ranked.back().first  * 1.05;
    double vmin = ranked.front().first * 0.98;
    double vr   = vmax - vmin;

    Svg svg(W, H);

    // horizontal gridlines
    for (int i = 0; i <= 5; ++i) {
        double v = vmin + vr * i / 5;
        double y = MT + ph - ((v - vmin) / vr) * ph;
        svg.line(ML, y, ML + pw, y, "#e0e0e0", 1, "4,3");
        std::ostringstream s; s << std::fixed << std::setprecision(0) << v;
        svg.text(ML - 6, y + 4, s.str(), "end", 10, "#666");
    }
    svg.line(ML, MT,      ML,      MT + ph, "#888", 1.5);
    svg.line(ML, MT + ph, ML + pw, MT + ph, "#888", 1.5);

    int n = (int)ranked.size();
    double bw = pw / n;
    for (int i = 0; i < n; ++i) {
        double v  = ranked[i].first;
        double bh = ((v - vmin) / vr) * ph;
        double bx = ML + i * bw;
        svg.rect(bx, MT + ph - bh, bw - 1, bh, (i == 0) ? "#e63946" : "#b0c4de");
    }

    // annotate best value above rank-1 bar
    double best_v = ranked[0].first;
    double best_y = MT + ph - ((best_v - vmin) / vr) * ph;
    std::ostringstream ann; ann << std::fixed << std::setprecision(0) << best_v;
    svg.text(ML + bw / 2, best_y - 6, ann.str(), "middle", 10, "#e63946");

    svg.text(ML + pw / 2, MT - 30,
             A + " – 100 configs ranked by avg (budget=100k, cap41_ss)",
             "middle", 14, "#222");
    svg.text(ML + pw / 2, MT + ph + 40, "Rank (best → worst)", "middle", 11, "#555");
    svg.text(ML - 65, MT + ph / 2, "Avg objective value", "middle", 11, "#555", -90);

    svg.save(out);
}

// ── per-parameter line chart (style: plot_results::line_chart) ─────────────────
// X = parameter value (sorted), two lines: avg (algorithm color) + best (red).
// Y zoomed to actual data range. Clips outliers above 90th-pct of avg.

void param_chart(const fs::path& out,
                 const std::string& title,
                 const std::string& xlabel,
                 std::vector<std::pair<double,double>> avg_pts,
                 std::vector<std::pair<double,double>> best_pts,
                 const std::string& avg_col,
                 const std::string& best_col = "#e63946")
{
    if (avg_pts.empty()) return;

    const int    W  = 900, H  = 480;
    const double ML = 90, MR = 160, MT = 70, MB = 60;
    const double pw = W - ML - MR;
    const double ph = H - MT - MB;
    const double x0 = ML, x1 = ML + pw, y0 = MT, y1 = MT + ph;

    // sort by parameter value
    std::vector<int> idx(avg_pts.size()); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){
        return avg_pts[a].first < avg_pts[b].first;
    });

    // X range
    double xmin = avg_pts[idx.front()].first, xmax = avg_pts[idx.back()].first;
    double xr = xmax - xmin; if (xr < 1e-9) xr = 1;
    xmin -= xr*0.02; xmax += xr*0.02; xr = xmax - xmin;

    // Y range: clip above 90th-pct of avg to suppress outliers
    std::vector<double> all_avg; for (auto& [x,y] : avg_pts) all_avg.push_back(y);
    std::sort(all_avg.begin(), all_avg.end());
    double y_ceil = all_avg[(int)(all_avg.size() * 0.90)];

    double y_floor = best_pts[0].second;
    for (auto& [x,y] : best_pts) y_floor = std::min(y_floor, y);

    int clipped = 0;
    for (auto v : all_avg) if (v > y_ceil) ++clipped;

    double yr = y_ceil - y_floor; if (yr < 1) yr = 1;
    double ymin = y_floor - yr * 0.05;
    double ymax = y_ceil  + yr * 0.10;
    yr = ymax - ymin;

    auto px  = [&](double x) { return x0 + (x - xmin) / xr * pw; };
    auto pyc = [&](double y) { return std::max(y0, std::min(y1, y0 + ph - (y-ymin)/yr*ph)); };

    Svg svg(W, H);

    // vertical grid
    for (int i = 0; i <= 5; ++i) {
        double xv = xmin + xr*i/5, xp = px(xv);
        svg.line(xp, y0, xp, y1, "#e0e0e0", 1, "4,3");
        std::ostringstream s;
        if      (xr > 5000) s << std::fixed << std::setprecision(0) << xv;
        else if (xr > 1)    s << std::fixed << std::setprecision(2) << xv;
        else                s << std::fixed << std::setprecision(4) << xv;
        svg.text(xp, y1 + 16, s.str(), "middle", 10, "#666");
    }
    // horizontal grid
    for (int i = 0; i <= 5; ++i) {
        double yv = ymin + yr*i/5, yp = pyc(yv);
        svg.line(x0, yp, x1, yp, "#e0e0e0", 1, "4,3");
        std::ostringstream s; s << std::fixed << std::setprecision(0) << yv;
        svg.text(x0 - 6, yp + 4, s.str(), "end", 10, "#666");
    }
    svg.line(x0, y0, x0, y1, "#888", 1.5);
    svg.line(x0, y1, x1, y1, "#888", 1.5);

    // avg polyline
    {
        std::vector<std::pair<double,double>> pts;
        for (int i : idx) pts.push_back({px(avg_pts[i].first), pyc(avg_pts[i].second)});
        svg.polyline(pts, avg_col, 2.0);
    }
    // best polyline
    {
        std::vector<std::pair<double,double>> pts;
        for (int i : idx) pts.push_back({px(best_pts[i].first), pyc(best_pts[i].second)});
        svg.polyline(pts, best_col, 2.0);
    }

    // legend in right margin (style: plot_results::line_chart)
    double lx = x1 + 15, ly = y0 + 20;
    svg.line(lx,      ly + 7,  lx + 20, ly + 7,  best_col, 2.5);
    svg.text(lx + 25, ly + 11, "best",          "start", 10, "#444");
    svg.line(lx,      ly + 29, lx + 20, ly + 29, avg_col,  2.5);
    svg.text(lx + 25, ly + 33, "avg (10 runs)", "start", 10, "#444");

    svg.text(x0 + pw/2, MT - 30, title,              "middle", 14, "#222");
    svg.text(x0 + pw/2, MT + ph + 42, xlabel,        "middle", 11, "#555");
    svg.text(ML - 65,   y0 + ph/2, "Avg objective value", "middle", 11, "#555", -90);

    if (clipped > 0) {
        std::ostringstream s; s << clipped << " outlier(s) not shown";
        svg.text(x1 - 2, y0 + 14, s.str(), "end", 9, "#aaa");
    }

    svg.save(out);
}

// ── generate per-parameter charts for one CSV ─────────────────────────────────

void gen_params(const fs::path& csv,
                const std::string& algo,
                const std::vector<std::pair<int,std::string>>& params,
                int avg_col, int best_col,
                const std::string& avg_color)
{
    auto rows = read_rows(csv);
    if (rows.empty()) return;
    std::string A = algo; for (auto& c : A) c = std::toupper(c);

    for (auto& [ci, name] : params) {
        std::vector<std::pair<double,double>> avg_pts, best_pts;
        for (auto& r : rows) {
            avg_pts .push_back({dcol(r,ci), dcol(r,avg_col)});
            best_pts.push_back({dcol(r,ci), dcol(r,best_col)});
        }
        param_chart(PLOTS / (algo + "_" + name + ".svg"),
                    A + ": " + name + " (budget=100k, cap41_ss)",
                    name, avg_pts, best_pts, avg_color);
    }
}

int main() {
    fs::create_directories(PLOTS);

    // EA: col1=pop, 2=gen_actual, 3=tour, 4=pm, 5=px | best=9, avg=10
    std::cout << "EA:\n";
    {
        auto rows = read_rows(DATA2 / "ea_budget.csv");
        ranked_chart(PLOTS / "ea_ranked.svg", "ea", rows, 10, 9);
    }
    gen_params(DATA2/"ea_budget.csv", "ea",
               {{1,"pop"},{2,"gen_actual"},{3,"tour"},{4,"pm"},{5,"px"}},
               10, 9, "#f4a261");

    // SA: col1=T0, 2=cool, 4=ipt | best=7, avg=8
    std::cout << "SA:\n";
    {
        auto rows = read_rows(DATA2 / "sa_budget.csv");
        ranked_chart(PLOTS / "sa_ranked.svg", "sa", rows, 8, 7);
    }
    gen_params(DATA2/"sa_budget.csv", "sa",
               {{1,"T0"},{2,"cool"},{4,"ipt"}},
               8, 7, "#e9c46a");

    // VNS: col1=iter, 2=shake, 3=ls_actual | best=6, avg=7
    std::cout << "VNS:\n";
    {
        auto rows = read_rows(DATA2 / "vns_budget.csv");
        ranked_chart(PLOTS / "vns_ranked.svg", "vns", rows, 7, 6);
    }
    gen_params(DATA2/"vns_budget.csv", "vns",
               {{1,"iter"},{2,"shake"},{3,"ls_actual"}},
               7, 6, "#2a9d8f");

    // GRASP: col1=iter, 2=rcl, 3=ls_actual | best=6, avg=7
    std::cout << "GRASP:\n";
    {
        auto rows = read_rows(DATA2 / "grasp_budget.csv");
        ranked_chart(PLOTS / "grasp_ranked.svg", "grasp", rows, 7, 6);
    }
    gen_params(DATA2/"grasp_budget.csv", "grasp",
               {{1,"iter"},{2,"rcl"},{3,"ls_actual"}},
               7, 6, "#8ecae6");

    std::cout << "Done. " << PLOTS.string() << "\n";
}
