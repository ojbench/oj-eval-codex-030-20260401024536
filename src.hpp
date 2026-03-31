// Heuristic digit classifier for 28x28 grayscale images (values in [0,1])
// Implements int judge(IMAGE_T &img) where IMAGE_T = std::vector<std::vector<double>>

#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

typedef std::vector<std::vector<double>> IMAGE_T;

namespace nr_heur {

struct BBox { int r0, c0, r1, c1; }; // inclusive bounds

static inline int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }

static void otsu_threshold(const IMAGE_T &img, double &thr) {
    // Compute Otsu threshold on [0,1]
    const int bins = 32;
    int h[bins];
    std::fill(h, h+bins, 0);
    int n = 0;
    for (const auto &row : img) for (double v : row) {
        int b = (int)std::floor(std::clamp(v, 0.0, 1.0) * (bins-1) + 0.5);
        b = std::clamp(b, 0, bins-1);
        h[b]++; n++;
    }
    if (n == 0) { thr = 0.5; return; }
    double sum = 0.0; for (int i=0;i<bins;i++) sum += i * h[i];
    double sumB = 0.0; int wB = 0; double maxVar = -1.0; int bestT = bins/2;
    for (int t=0;t<bins;t++) {
        wB += h[t];
        if (wB == 0) continue;
        int wF = n - wB; if (wF == 0) break;
        sumB += t * h[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;
        double varBetween = (double)wB * (double)wF * (mB - mF) * (mB - mF);
        if (varBetween > maxVar) { maxVar = varBetween; bestT = t; }
    }
    thr = (double)bestT / (bins-1);
}

static std::vector<std::vector<unsigned char>> binarize(const IMAGE_T &img) {
    int H = (int)img.size();
    int W = H? (int)img[0].size() : 0;
    double thr; otsu_threshold(img, thr);
    // Because digits are white on black, threshold above thr as foreground
    std::vector<std::vector<unsigned char>> b(H, std::vector<unsigned char>(W, 0));
    for (int r=0;r<H;r++) {
        for (int c=0;c<W;c++) {
            b[r][c] = (img[r][c] >= thr ? 1: 0);
        }
    }
    return b;
}

static bool bbox(const std::vector<std::vector<unsigned char>> &b, BBox &bb) {
    int H = (int)b.size(); if (!H) return false; int W = (int)b[0].size();
    int r0=H, c0=W, r1=-1, c1=-1;
    for (int r=0;r<H;r++) for (int c=0;c<W;c++) if (b[r][c]) {
        r0 = std::min(r0, r); c0 = std::min(c0, c);
        r1 = std::max(r1, r); c1 = std::max(c1, c);
    }
    if (r1<r0 || c1<c0) return false;
    bb = {r0,c0,r1,c1};
    return true;
}

static int count_segments_row(const std::vector<unsigned char> &row, int c0, int c1) {
    int segs=0; int c=c0; int prev=0;
    for (; c<=c1; ++c) {
        int v = row[c];
        if (v==1 && prev==0) segs++;
        prev = v;
    }
    return segs;
}

static void flood_fill(std::vector<std::vector<unsigned char>> &m, int sr, int sc, unsigned char target, unsigned char repl){
    int H=(int)m.size(), W=(int)m[0].size();
    if (sr<0||sr>=H||sc<0||sc>=W) return;
    if (m[sr][sc]!=target) return;
    std::queue<std::pair<int,int>>q; q.push({sr,sc}); m[sr][sc]=repl;
    const int dr[4]={1,-1,0,0}; const int dc[4]={0,0,1,-1};
    while(!q.empty()){
        auto [r,c]=q.front(); q.pop();
        for(int k=0;k<4;k++){
            int nr=r+dr[k], nc=c+dc[k];
            if(nr>=0&&nr<H&&nc>=0&&nc<W&&m[nr][nc]==target){ m[nr][nc]=repl; q.push({nr,nc}); }
        }
    }
}

static int count_holes_and_centroid(const std::vector<std::vector<unsigned char>> &bin, const BBox &bb, double &hy, double &hx) {
    // Count holes by flood fill on background within a padded bbox
    int r0=bb.r0, c0=bb.c0, r1=bb.r1, c1=bb.c1;
    int H = r1 - r0 + 1, W = c1 - c0 + 1;
    std::vector<std::vector<unsigned char>> m(H+2, std::vector<unsigned char>(W+2, 0));
    for(int r=0;r<H;r++) for(int c=0;c<W;c++) m[r+1][c+1] = bin[r0+r][c0+c]? 0 : 1; // background=1
    // Remove outer background connected to border
    flood_fill(m, 0, 0, 1, 2);
    int holes=0; double sy=0.0,sx=0.0;
    for(int r=0;r<H+2;r++){
        for(int c=0;c<W+2;c++){
            if(m[r][c]==1){
                // new hole
                holes++;
                // compute centroid by flood fill to 3
                std::queue<std::pair<int,int>>q; q.push({r,c}); m[r][c]=3;
                int cnt=0; double csx=0.0,csy=0.0;
                const int dr[4]={1,-1,0,0}; const int dc[4]={0,0,1,-1};
                while(!q.empty()){
                    auto [yr, xc]=q.front(); q.pop();
                    cnt++; csy += yr; csx += xc;
                    for(int k=0;k<4;k++){
                        int ny=yr+dr[k], nx=xc+dc[k];
                        if(ny>=0&&ny<H+2&&nx>=0&&nx<W+2&&m[ny][nx]==1){ m[ny][nx]=3; q.push({ny,nx}); }
                    }
                }
                if(cnt>0){ sy += csy / cnt; sx += csx / cnt; }
            }
        }
    }
    if (holes>0){ hy = (sy/holes) / (H+2); hx = (sx/holes) / (W+2); }
    else { hy=0.5; hx=0.5; }
    return holes;
}

static void projections(const std::vector<std::vector<unsigned char>> &bin, const BBox &bb,
                        std::vector<int> &prow, std::vector<int> &pcol) {
    int r0=bb.r0, c0=bb.c0, r1=bb.r1, c1=bb.c1;
    int H = r1 - r0 + 1, W = c1 - c0 + 1;
    prow.assign(H,0); pcol.assign(W,0);
    for(int r=0;r<H;r++){
        int sum=0; for(int c=0;c<W;c++){ if(bin[r0+r][c0+c]){ sum++; pcol[c]++; } }
        prow[r]=sum;
    }
}

static void mass_props(const std::vector<std::vector<unsigned char>> &bin, const BBox &bb,
                       double &cy, double &cx, double &fr) {
    int r0=bb.r0, c0=bb.c0, r1=bb.r1, c1=bb.c1;
    int H = r1 - r0 + 1, W = c1 - c0 + 1;
    double sy=0.0,sx=0.0; int cnt=0;
    for(int r=0;r<H;r++) for(int c=0;c<W;c++) if(bin[r0+r][c0+c]){ sy += r; sx += c; cnt++; }
    fr = (double)cnt / (double)(H*W);
    if(cnt>0){ cy = sy / H; cx = sx / W; cy/= (double)H; cx/= (double)W; /* normalized [0,1) */ }
    else { cy = 0.5; cx = 0.5; }
}

static double symmetry_score(const std::vector<std::vector<unsigned char>> &bin, const BBox &bb, bool vertical){
    int r0=bb.r0, c0=bb.c0, r1=bb.r1, c1=bb.c1;
    int H = r1 - r0 + 1, W = c1 - c0 + 1;
    int matches=0, total=0;
    if(vertical){
        for(int r=0;r<H;r++){
            for(int c=0;c<W/2;c++){
                if (bin[r0+r][c0+c] == bin[r0+r][c0+(W-1-c)]) matches++;
                total++;
            }
        }
    } else {
        for(int r=0;r<H/2;r++){
            for(int c=0;c<W;c++){
                if (bin[r0+r][c0+c] == bin[r0+(H-1-r)][c0+c]) matches++;
                total++;
            }
        }
    }
    if(total==0) return 0.0;
    return (double)matches / (double)total;
}

static int classify(const std::vector<std::vector<unsigned char>> &bin, const BBox &bb){
    int r0=bb.r0, c0=bb.c0, r1=bb.r1, c1=bb.c1;
    int H = r1 - r0 + 1, W = c1 - c0 + 1;

    // Basic features
    double cy,cx,fr; mass_props(bin, bb, cy, cx, fr);
    double hy,hx; int holes = count_holes_and_centroid(bin, bb, hy, hx);
    std::vector<int> prow, pcol; projections(bin, bb, prow, pcol);

    auto avg_segments_rows = [&](int y0, int y1){
        y0 = clampi(y0, 0, H-1); y1 = clampi(y1, 0, H-1);
        if (y1 < y0) std::swap(y0, y1);
        double s=0.0; int cnt=0;
        for (int r=y0; r<=y1; ++r){ s += count_segments_row(bin[r0+r], c0, c1); cnt++; }
        return cnt? (s/cnt) : 0.0;
    };

    auto run_max_ratio = [&](int ry0, int ry1){
        ry0 = clampi(ry0, 0, H-1); ry1 = clampi(ry1, 0, H-1);
        if (ry1<ry0) std::swap(ry0, ry1);
        int maxRun=0;
        for(int r=ry0;r<=ry1;r++){
            int run=0; int best=0; int prev=0;
            for(int c=c0;c<=c1;c++){
                int v=bin[r0+r][c];
                if(v){ run++; best = std::max(best, run); }
                else run=0;
            }
            maxRun = std::max(maxRun, best);
        }
        return (double)maxRun / (double)std::max(1, W);
    };

    auto density_region = [&](int ry0, int ry1, int cx0, int cx1){
        ry0 = clampi(ry0, 0, H-1); ry1 = clampi(ry1, 0, H-1);
        if (ry1<ry0) std::swap(ry0, ry1);
        cx0 = clampi(cx0, 0, W-1); cx1 = clampi(cx1, 0, W-1);
        if (cx1<cx0) std::swap(cx0, cx1);
        int cnt=0, tot=(ry1-ry0+1)*(cx1-cx0+1);
        for(int r=ry0;r<=ry1;r++) for(int c=cx0;c<=cx1;c++) if(bin[r0+r][c0+c]) cnt++;
        return tot? (double)cnt/(double)tot : 0.0;
    };

    if (holes >= 2) return 8;
    if (holes == 1){
        // Use hole vertical position
        if (hy > 0.6) return 6;
        // Distinguish 0 vs 9 using bottom row segment patterns and symmetry
        double botSeg = avg_segments_rows((int)(H*0.7), H-1);
        double sv = symmetry_score(bin, bb, true);
        double sh = symmetry_score(bin, bb, false);
        if (sv > 0.85 && sh > 0.8) return 0;
        if (botSeg < 1.5) return 9;
        return 0;
    }

    // No holes: likely 1,2,3,4,5,7
    double whRatio = (double)W / (double)H;
    // Thin vertical -> 1
    if (whRatio < 0.5 || fr < 0.14) {
        // Check central column continuity
        int col = c0 + W/2; int rowsWith=0;
        for(int r=0;r<H;r++) if(bin[r0+r][col]) rowsWith++;
        if ((double)rowsWith / H > 0.6) return 1;
    }

    // Strong top bar and sparse bottom -> 7
    double topRun = run_max_ratio(0, std::max(0,(int)(H*0.2)));
    double lowDense = density_region((int)(H*0.6), H-1, 0, W-1);
    if (topRun > 0.7 && lowDense < 0.22 && cy < 0.5) return 7;

    // Detect open 4: strong right vertical stroke and mid horizontal
    int rightCols = std::max(1, W/6);
    int rowsWithRight=0; for(int r=0;r<H;r++){
        bool any=false; for(int c=W-rightCols;c<W;c++) if(bin[r0+r][c0+c]){ any=true; break; }
        if(any) rowsWithRight++;
    }
    double rightContinuity = (double)rowsWithRight / H;
    double midRun = run_max_ratio(std::max(0,(int)(H*0.45)), std::min(H-1,(int)(H*0.55)));
    if (rightContinuity > 0.8 && midRun > 0.5) return 4;

    // Bottom long run -> 2 or 5
    double bottomRun = run_max_ratio(std::max(0,(int)(H*0.75)), H-1);
    if (bottomRun > 0.7){
        double ll = density_region((int)(H*0.6), H-1, 0, W/2-1);
        double lr = density_region((int)(H*0.6), H-1, W/2, W-1);
        if (ll > lr * 1.2) return 2;
        return 5;
    }

    // Right-heavy -> 3
    double leftAll = density_region(0, H-1, 0, W/2-1);
    double rightAll = density_region(0, H-1, W/2, W-1);
    if (rightAll > leftAll * 1.35) return 3;

    // Fallbacks: choose among 2 and 5 based on quadrant densities
    double ul = density_region(0, H/2-1, 0, W/2-1);
    double ur = density_region(0, H/2-1, W/2, W-1);
    double ll = density_region(H/2, H-1, 0, W/2-1);
    double lr = density_region(H/2, H-1, W/2, W-1);
    if (ul > ur && ll > lr) return 5; // more on left
    return 2;
}

} // namespace nr_heur

int judge(IMAGE_T &img) {
    using namespace nr_heur;
    if (img.empty() || img[0].empty()) return 1;
    auto bin = binarize(img);
    BBox bb;
    if (!bbox(bin, bb)) return 1;
    int ans = classify(bin, bb);
    if (ans < 0 || ans > 9) ans = 0;
    return ans;
}

