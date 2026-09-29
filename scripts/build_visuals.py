from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "generated"
PAGES = ROOT / "tmp" / "pdf_pages"
PAGES.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

for font in ["Malgun Gothic", "AppleGothic", "DejaVu Sans"]:
    if font in {f.name for f in mpl.font_manager.fontManager.ttflist}:
        mpl.rcParams["font.family"] = font
        break
mpl.rcParams["axes.unicode_minus"] = False

CMAP = LinearSegmentedColormap.from_list("pca_div", ["#2166AC", "#F7F7F7", "#B2182B"])
CMAP.set_bad("#D7DDE1")
METRICS = ["rank_ic", "quintile_ic", "delta_rank_ic", "delta_quintile_ic"]
LABELS = {
    "rank_ic": "Rank IC",
    "quintile_ic": "Quintile IC (분위수 번호 1-5 상관계수)",
    "delta_rank_ic": "Delta Rank IC (PCA - Raw)",
    "delta_quintile_ic": "Delta Quintile IC (PCA - Raw)",
}


def arrays():
    cases = pd.read_csv(DATA / "cases.csv")
    full = pd.read_csv(DATA / "full_summary.csv")
    annual = pd.read_csv(DATA / "annual_summary.csv")
    years = sorted(annual.year.unique().astype(int))
    ks = [5, 10, 20, 60]
    av, aq, fv, fq = {}, {}, {}, {}
    for metric in METRICS:
        a = np.full((len(cases), len(ks), len(years)), np.nan)
        q = np.full_like(a, np.nan)
        f = np.full((len(cases), len(ks)), np.nan)
        fqq = np.full_like(f, np.nan)
        for ci, case in enumerate(cases.case_id):
            for ki, k in enumerate(ks):
                fr = full[(full.metric == metric) & (full.case_id == case) & (full.k == k)]
                if not fr.empty:
                    f[ci, ki] = fr.iloc[0]["mean"]
                    fqq[ci, ki] = fr.iloc[0].q_pos
                for yi, year in enumerate(years):
                    ar = annual[(annual.metric == metric) & (annual.case_id == case) & (annual.k == k) & (annual.year == year)]
                    if not ar.empty:
                        a[ci, ki, yi] = ar.iloc[0]["mean"]
                        q[ci, ki, yi] = ar.iloc[0].q_pos_global_space
        av[metric], aq[metric], fv[metric], fq[metric] = a, q, f, fqq
    all_values = np.concatenate([
        np.abs(av[metric][np.isfinite(av[metric])]) for metric in METRICS
    ])
    common_limit = max(float(np.quantile(all_values, .98)), 1e-6)
    limits = {metric: common_limit for metric in METRICS}
    return cases, full, annual, years, ks, av, aq, fv, fq, limits


def add_heatmap(ax, values, qvals, title, case_ids, ks, limit, percent=False):
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=CMAP, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
    ax.set_xticks(range(len(ks)), [str(k) for k in ks], fontsize=7)
    ax.set_xlabel("미래기간 k", fontsize=8)
    ax.set_yticks(range(len(case_ids)), case_ids, fontsize=4.3)
    ax.set_ylabel("CaseID", fontsize=8)
    ax.set_xticks(np.arange(-.5, len(ks), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(case_ids), 1), minor=True)
    ax.grid(which="minor", color="#81909A", linewidth=.22)
    ax.tick_params(which="minor", bottom=False, left=False)
    for y in [3.5, 15.5, 27.5]:
        ax.axhline(y, color="#263A46", lw=1.1)
    sig = np.argwhere(np.isfinite(qvals) & (qvals < .05))
    if sig.size:
        ax.scatter(sig[:,1], sig[:,0], s=4.5, c="#111111", marker="o", linewidths=0)
    cb = plt.colorbar(im, ax=ax, fraction=.026, pad=.02)
    cb.ax.tick_params(labelsize=6)
    if percent:
        cb.formatter = mpl.ticker.PercentFormatter(xmax=1, decimals=1); cb.update_ticks()


def make_pages(cases, full, annual, years, ks, av, aq, fv, fq, limits):
    # Cover
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    ax = fig.add_axes([0,0,1,1]); ax.axis("off")
    ax.text(.06,.86,"PCA Rank Validation",fontsize=30,fontweight="bold",color="#153B5B")
    ax.text(.06,.80,"전체 160개 조합의 Rank IC · Quintile IC · Delta 검정",fontsize=16,color="#526773")
    ax.text(.06,.70,"공간 정의",fontsize=16,fontweight="bold",color="#153B5B")
    ax.text(.08,.64,"x = 미래기간 k (5, 10, 20, 60)\ny = 40개 CaseID (W · n · PC)\nz = 연도 t\n색 = 효과, 점 = global-space BH q < 0.05",fontsize=13,linespacing=1.7)
    sig = full.groupby("metric").apply(lambda x:int((x.q_pos<.05).sum()))
    y=.43
    for m in METRICS:
        ax.text(.08,y,f"{LABELS[m]}: {sig.get(m,0)} / {len(full[full.metric==m])} 조합 유의",fontsize=13)
        y-=.055
    ax.text(.06,.10,"청색 = 음수/악화 · 백색 = 0 · 적색 = 양수/개선 · 회색 = N/A",fontsize=12,color="#526773")
    fig.savefig(PAGES/"page-01-cover.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # Full sample
    fig, axes = plt.subplots(2,2,figsize=(16,12),constrained_layout=True)
    for ax,m in zip(axes.flat,METRICS):
        add_heatmap(ax,fv[m],fq[m],LABELS[m],cases.case_id,ks,limits[m])
    fig.suptitle("전체기간 평면 히트맵 | 검은 점: 방향성 BH q < 0.05",fontsize=16,fontweight="bold",color="#153B5B")
    fig.savefig(PAGES/"page-02-full-heatmaps.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # Annual pages
    for yi,year in enumerate(years):
        fig, axes = plt.subplots(2,2,figsize=(16,12),constrained_layout=True)
        for ax,m in zip(axes.flat,METRICS):
            add_heatmap(ax,av[m][:,:,yi],aq[m][:,:,yi],LABELS[m],cases.case_id,ks,limits[m])
        fig.suptitle(f"{year}년 공간 평면 | 전 연도 고정 색상범위 · global-space BH",fontsize=16,fontweight="bold",color="#153B5B")
        fig.savefig(PAGES/f"page-{yi+3:02d}-year-{year}.png",dpi=160,bbox_inches="tight"); plt.close(fig)

    # Significance count through time
    fig, axes = plt.subplots(2,1,figsize=(16,9),constrained_layout=True)
    for m in ["rank_ic","quintile_ic"]:
        counts=[int(((annual.metric==m)&(annual.year==yr)&(annual.q_pos_global_space<.05)).sum()) for yr in years]
        axes[0].plot(years,counts,marker="o",label=LABELS[m])
    axes[0].set_title("원 지표: 연도별 global-space BH 유의 조합 수",loc="left",fontweight="bold")
    axes[0].legend();axes[0].grid(alpha=.25);axes[0].set_ylabel("조합 수")
    for m in ["delta_rank_ic","delta_quintile_ic"]:
        counts=[int(((annual.metric==m)&(annual.year==yr)&(annual.q_pos_global_space<.05)).sum()) for yr in years]
        axes[1].plot(years,counts,marker="o",label=LABELS[m])
    axes[1].set_title("Delta: 연도별 global-space BH 유의 조합 수",loc="left",fontweight="bold")
    axes[1].legend();axes[1].grid(alpha=.25);axes[1].set_ylabel("조합 수");axes[1].set_xlabel("연도")
    fig.savefig(PAGES/f"page-{len(years)+3:02d}-significance.png",dpi=160,bbox_inches="tight");plt.close(fig)


def draw_space(
    ax,
    values,
    qvals,
    years,
    ks,
    metric,
    limit,
    spacing=1.5,
    selected=None,
    background_alpha=.055,
    selected_alpha=.98,
    significant_alpha=None,
    edge_alpha=.22,
    edge_width=.16,
    outline_significant=False,
    layer_alphas=None,
):
    if selected is None: selected=len(years)-1
    polys, colors, sig_polys = [], [], []
    norm = TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    is_delta=metric.startswith("delta_")
    case_indices=range(4,values.shape[0]) if is_delta else range(values.shape[0])
    for yi in range(len(years)):
        z=yi*spacing
        for ci in case_indices:
            plot_ci=ci-4 if is_delta else ci
            for ki in range(values.shape[1]):
                v=values[ci,ki,yi]
                if not np.isfinite(v): continue
                x0,x1=ki+.56,ki+1.44;y0,y1=plot_ci+.56,plot_ci+1.44
                poly=[(x0,y0,z),(x1,y0,z),(x1,y1,z),(x0,y1,z)]
                polys.append(poly)
                is_sig=np.isfinite(qvals[ci,ki,yi]) and qvals[ci,ki,yi]<.05
                if outline_significant and is_sig:
                    sig_polys.append(poly)
                if layer_alphas is not None:
                    a=float(layer_alphas[yi])
                else:
                    a=selected_alpha if yi==selected else background_alpha
                    if significant_alpha is not None and is_sig and yi!=selected:
                        a=significant_alpha
                c=list(CMAP(norm(v)));c[3]=a;colors.append(c)
    coll=Poly3DCollection(polys,facecolors=colors,edgecolors=(.25,.30,.33,edge_alpha),linewidths=edge_width)
    ax.add_collection3d(coll)
    if sig_polys:
        sig_coll=Poly3DCollection(
            sig_polys,
            facecolors=(0,0,0,0),
            edgecolors=(.04,.04,.04,.72),
            linewidths=.48,
        )
        ax.add_collection3d(sig_coll)
    ax.set_xlim(.4,4.6)
    if is_delta:
        ax.set_ylim(.4,36.6)
        ax.set_yticks([6.5,18.5,30.5],["W60","W120","W252"])
    else:
        ax.set_ylim(.4,40.6)
        ax.set_yticks([2.5,10.5,22.5,34.5],["Raw","W60","W120","W252"])
    ax.set_zlim(-.2,max(1,(len(years)-1)*spacing+.2))
    ax.set_xticks(range(1,5),ks)
    zidx=list(range(0,len(years),2));ax.set_zticks([i*spacing for i in zidx],[years[i] for i in zidx])
    ax.set_xlabel("미래기간 k",labelpad=10)
    ax.set_ylabel("Case block (W · PC · n)",labelpad=22)
    ax.set_zlabel("연도",labelpad=14)
    ax.tick_params(axis="x",pad=2,labelsize=8)
    ax.tick_params(axis="y",pad=7,labelsize=8)
    ax.tick_params(axis="z",pad=4,labelsize=8)
    ax.set_title(f"{LABELS[metric]} | selected {years[selected]}",fontsize=10,fontweight="bold")
    ax.view_init(elev=24,azim=-52);ax.set_box_aspect((1.1,3.2,2.0));ax.grid(False)


def make_space_overview(years,ks,av,aq,limits):
    fig=plt.figure(figsize=(18,12),facecolor="white")
    for idx,m in enumerate(METRICS,1):
        ax=fig.add_subplot(2,2,idx,projection="3d")
        draw_space(ax,av[m],aq[m],years,ks,m,limits[m],2.4,len(years)-1,
                   background_alpha=.32,selected_alpha=.96,edge_alpha=.11,edge_width=.10)
        if idx <= 2:
            ax.set_xlabel("")
        sm=mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-limits[m],vcenter=0,vmax=limits[m]),cmap=CMAP)
        cb=fig.colorbar(sm,ax=ax,shrink=.48,aspect=18,pad=.015)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("연도별 평면 적층 공간 | 네 지표 공통 색상범위",fontsize=16,fontweight="bold",color="#153B5B")
    fig.subplots_adjust(left=.02,right=.98,bottom=.055,top=.94,wspace=.01,hspace=.18)
    out=OUT/"05_PCA_RANK_SPACE_OVERVIEW.png";fig.savefig(out,dpi=180,bbox_inches="tight");plt.close(fig)
    return out


def make_clear_space_pngs(years,ks,av,aq,limits):
    """Readable static exports: every annual layer remains visible."""
    selected=len(years)-1
    common=dict(
        spacing=2.4,
        selected=selected,
        background_alpha=.32,
        selected_alpha=.96,
        significant_alpha=None,
        edge_alpha=.11,
        edge_width=.10,
        outline_significant=False,
    )

    summary=OUT/"09_PCA_RANK_SPACE_OVERVIEW_CLEAR.png"
    fig=plt.figure(figsize=(18,12),facecolor="white")
    for idx,m in enumerate(METRICS,1):
        ax=fig.add_subplot(2,2,idx,projection="3d")
        draw_space(ax,av[m],aq[m],years,ks,m,limits[m],**common)
        if idx <= 2:
            ax.set_xlabel("")
        sm=mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-limits[m],vcenter=0,vmax=limits[m]),cmap=CMAP)
        cb=fig.colorbar(sm,ax=ax,shrink=.48,aspect=18,pad=.015)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(
        "PCA Rank Validation | 전체 연도 가시형 3D 공간 요약",
        fontsize=17,fontweight="bold",color="#153B5B",y=.985,
    )
    fig.text(
        .5,.012,
        "모든 중간 연도 불투명도 32% · 선택 연도 96% · Delta 공간은 Raw 행 제외",
        ha="center",fontsize=10,color="#526773",
    )
    fig.subplots_adjust(left=.02,right=.98,bottom=.055,top=.94,wspace=.01,hspace=.18)
    fig.savefig(summary,dpi=200,facecolor="white",bbox_inches="tight")
    plt.close(fig)

    individual=[]
    for idx,m in enumerate(METRICS,1):
        path=OUT/f"09_{idx}_{m.upper()}_SPACE_CLEAR.png"
        fig=plt.figure(figsize=(13.5,10.5),facecolor="white")
        ax=fig.add_subplot(111,projection="3d")
        draw_space(ax,av[m],aq[m],years,ks,m,limits[m],**common)
        sm=mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-limits[m],vcenter=0,vmax=limits[m]),cmap=CMAP)
        cb=fig.colorbar(sm,ax=ax,shrink=.62,aspect=24,pad=.015)
        cb.ax.tick_params(labelsize=9)
        ax.set_title(f"{LABELS[m]} | 전체 연도 3D 공간",fontsize=15,fontweight="bold",pad=18)
        fig.text(
            .5,.025,
            "중간 연도 32% · 선택 연도 96% · Delta 공간은 Raw 행 제외 · 전 연도 공통 색상범위",
            ha="center",fontsize=10,color="#526773",
        )
        fig.subplots_adjust(left=.01,right=.99,bottom=.065,top=.93)
        fig.savefig(path,dpi=220,facecolor="white",bbox_inches="tight")
        plt.close(fig)
        individual.append(path)
    return summary, individual


def make_metric_time_videos(years,ks,av,aq,limits):
    """Create one smooth chronological selected-layer video per metric."""
    sys.path.insert(0,str(ROOT/"tmp"/"pydeps"))
    import imageio_ffmpeg

    outputs=[]
    fps=10
    hold_frames=3
    transition_frames=4
    background_alpha=.32
    selected_alpha=.96
    spacing=2.4
    width,height=1280,720

    for idx,m in enumerate(METRICS,1):
        out=OUT/f"10_{idx}_{m.upper()}_TIME_SELECTION.mp4"
        fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor="white")
        ax=fig.add_subplot(111,projection="3d")
        sm=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-limits[m],limits[m]),cmap=CMAP)
        sm.set_array([])
        cb=fig.colorbar(sm,ax=ax,shrink=.66,aspect=24,pad=.02)
        cb.set_label("연도별 평균 상관계수",fontsize=10)
        header=fig.suptitle("",fontsize=16,fontweight="bold",color="#153B5B",y=.965)
        footer=fig.text(
            .5,.025,
            "중간 연도 32% · 선택층 96% · 네 지표 공통 색상범위" +
            (" · Delta Raw 제외" if m.startswith("delta_") else ""),
            ha="center",fontsize=10,color="#526773",
        )
        writer=imageio_ffmpeg.write_frames(
            str(out),(width,height),fps=fps,codec="libx264",
            pix_fmt_in="rgb24",pix_fmt_out="yuv420p",quality=7,macro_block_size=2,
        )
        writer.send(None)

        def write_frame(selected,alphas,label):
            ax.clear()
            draw_space(
                ax,av[m],aq[m],years,ks,m,limits[m],spacing,selected,
                background_alpha=background_alpha,selected_alpha=selected_alpha,
                edge_alpha=.11,edge_width=.10,layer_alphas=alphas,
            )
            ax.set_title("")
            header.set_text(f"PCA Rank Validation | {LABELS[m]} | 선택 연도 {label}")
            fig.subplots_adjust(left=.02,right=.91,bottom=.08,top=.90)
            fig.canvas.draw()
            frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
            writer.send(frame.tobytes())

        try:
            for yi in range(len(years)):
                alphas=np.full(len(years),background_alpha,dtype=float)
                alphas[yi]=selected_alpha
                for _ in range(hold_frames):
                    write_frame(yi,alphas,str(years[yi]))
                if yi < len(years)-1:
                    for step in range(1,transition_frames+1):
                        t=step/transition_frames
                        blend=np.full(len(years),background_alpha,dtype=float)
                        blend[yi]=selected_alpha+(background_alpha-selected_alpha)*t
                        blend[yi+1]=background_alpha+(selected_alpha-background_alpha)*t
                        write_frame(yi+1,blend,f"{years[yi]} → {years[yi+1]}")
        finally:
            writer.close();plt.close(fig)
        outputs.append(out)
        print(f"Created {out.name}",flush=True)
    return outputs


def make_video(years,ks,av,aq,limits):
    sys.path.insert(0,str(ROOT/"tmp"/"pydeps"))
    import imageio_ffmpeg
    out=OUT/"06_PCA_RANK_SPACE_ANIMATION.mp4"
    fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor="white")
    ax=fig.add_subplot(111,projection="3d")
    width,height=1280,720
    writer=imageio_ffmpeg.write_frames(str(out),(width,height),fps=12,codec="libx264",pix_fmt_in="rgb24",pix_fmt_out="yuv420p",quality=7,macro_block_size=2)
    writer.send(None)
    try:
        sequence=[]
        for mi,m in enumerate(METRICS):
            if mi==0:
                sequence += [(m,g,len(years)-1) for g in np.linspace(.65,3.2,24)]
            sequence += [(m,3.2,yi) for yi in range(len(years)) for _ in range(2)]
            sequence += [(m,3.2,len(years)-1) for _ in range(8)]
        for m,g,yi in sequence:
            ax.clear();draw_space(ax,av[m],aq[m],years,ks,m,limits[m],float(g),yi)
            fig.suptitle("PCA Rank Validation - Heatmap Space",fontsize=17,fontweight="bold",color="#153B5B")
            fig.canvas.draw();frame=np.asarray(fig.canvas.buffer_rgba())[:,:,:3]
            writer.send(frame.tobytes())
    finally:
        writer.close();plt.close(fig)
    return out


def main():
    cases,full,annual,years,ks,av,aq,fv,fq,limits=arrays()
    make_pages(cases,full,annual,years,ks,av,aq,fv,fq,limits)
    make_space_overview(years,ks,av,aq,limits)
    make_video(years,ks,av,aq,limits)
    print("Visuals complete")


if __name__=="__main__":
    main()
