"""Figs. 2 and 3 of the paper (fig2.pdf, fig3.pdf) and the numbers it quotes from them: python3 figures.py"""
import numpy as np
from matplotlib import legend_handler as lh, patheffects as pe, pyplot as plt, transforms as mt
from scipy import ndimage, signal

x, N = np.arange(-400, 401, 8), 2 ** 15     # receiver positions (mm), FFT length > 7680 + 10000 + 7680 - 2 taps
chirp = np.load("data/chirp.npy")
Ge = np.fft.rfft(np.load("data/G_ej.npy"), N)
Gj = np.fft.rfft(np.concatenate([np.load("data/G_jr_1.npy"), np.load("data/G_jr_2.npy")]), N)
compress = lambda y: signal.fftconvolve(y, chirp[None, None, ::-1], axes=-1)   # pulse compression along time


def crosstalk(z, pairs):
    """Each emitter's energy at every receiver (dB re the loudest), mean away from the targets (dB re the weaker one)"""
    curves, means = [], []
    for e in range(2):
        ts = pairs[pairs[:, 1] == e, 0]
        n0 = np.abs(z[e, ts]).argmax() % z.shape[-1]                   # arrival at the louder target
        E = (z[e, :, n0 - 500:n0 + 500] ** 2).sum(-1)                  # within +-11.3 ms
        curves.append(10 * np.log10(E / E.max()))
        means.append(10 * np.log10(E[np.abs(np.arange(101)[:, None] - ts).min(1) > 2].mean() / E[ts].min()))
    return np.array(curves), np.array(means)


def focus(z, j):
    """Records z (receiver, t), target j: space-time map, zoom times (ms) and envelope (dB), -3 dB times (ms),
    [FWHM (us), medians (dB)]"""
    rms = lambda v: np.sqrt(ndimage.uniform_filter1d(v ** 2, 11))              # moving RMS over 0.25 ms
    n0 = np.abs(z[j]).argmax()
    field = rms(np.abs(signal.hilbert(z[:, n0 - 420:n0 + 420])))[:, 200:-200]  # margins for the Hilbert transform
    field = (20 * np.log10(field / field[j].max())).astype(np.float32)      # single precision, as published
    env = np.abs(signal.hilbert(signal.resample(z[j - 5:j + 6, n0 - 64:n0 + 65], 129 * 16, axis=-1)))
    e, p, h = env[5], env[5].argmax(), env[5].max() * 2 ** -0.5
    t = (np.arange(129 * 16) / 16 - 64) / 44.1                              # 44.1 samples per ms
    half = lambda i: t[i] + (h - e[i]) / (e[i + 1] - e[i]) * (t[i + 1] - t[i])  # crossing between i and i + 1
    lo, hi = half(p - np.argmax(e[p::-1] < h)), half(p + np.argmax(e[p:] < h) - 1)
    y = rms(z[j, n0 - 1113:n0 + 1113])[11:-11]                                 # record, not envelope, +-25 ms
    y, ty = 20 * np.log10(y / y.max()), np.arange(-1102, 1102) / 44.1
    return field, t[p - 105:p + 106], 20 * np.log10(env[:, p - 105:p + 106] / e[p]), lo, hi, [1000 * (hi - lo),
           np.median(field[np.abs(np.arange(101) - j) > 2]), np.median(y[ty < -2]), np.median(y[ty > 2])]


np.set_printoptions(precision=1, floatmode="fixed")
plt.rcParams.update({"font.family": "cmr10", "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
                     "axes.unicode_minus": False, "patch.linewidth": 0.5, "legend.fancybox": False})
MEAS, PRED, INK = ("#1f5fa8", "#c9531c"), ("#7fb1e6", "#f0a06f"), "#333333"   # emitters 1 and 2: dots, lines
BOX = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85)
GRAY = dict(cmap="gray_r", vmin=-40, vmax=0, origin="lower", aspect="auto", interpolation="bilinear")

for case, k, H, labels in (("single", 2, 4.8, [[(-104, -3.69), (-104, -9.02)], [(232, -13.66), (-256, -16.95)]]),
                           ("double", 3, 2.67, [[(104, -5.74), (-72, -6.52)], [(184, -10.73), (-200, -13.67)]])):
    npz, single = np.load(f"data/{case}.npz"), case == "single"
    pairs, r = npz["pairs"], {}
    for m in ("tr", "opt"):
        z = compress(npz[m + "_rec"])
        h = np.fft.irfft(np.einsum("ikf,jkf->ijf", Ge * np.fft.rfft(npz[m + "_filter"], N), Gj), N)  # h_{E_i->R_j}
        r[m] = crosstalk(z, pairs)
        r[m + "_pred"] = crosstalk(compress(signal.fftconvolve(h, chirp[None, None], axes=-1)), pairs)
        (cm, mm), (cp, mp) = r[m], r[m + "_pred"]
        print(f"Fig. {k} {m}: |meas. - pred.| {abs(mm - mp).max():.2f} dB (means), {abs(cm - cp).max():.2f} dB (all)")
        if single:
            r[m + "_focus"] = [focus(z[e], j) for j, e in pairs]
            r[m + "_num"] = np.array([f[-1] for f in r[m + "_focus"]]).T
            print(f"Fig. 2 {m} (emitters 1, 2): FWHM (us), median away, at target before, after (dB)\n{r[m + '_num']}")
    dB = r["opt"][1] - r["tr"][1]
    print(f"Fig. {k} opt - tr: mean crosstalk {dB} dB, energy factor {10 ** (-dB / 10)}")
    keys = {"measured": plt.Line2D([], [], ls="none", marker="o", ms=3.2, mfc="#52514e", mec="white", mew=0.35),
            "predicted": plt.Line2D([], [], color="#9a9994", lw=1.3)}                # legend
    if single:
        dn, j = r["opt_num"] - r["tr_num"], pairs[0, 0]
        print(f"Fig. 2 opt - tr: median away from the target {dn[1]} dB, at the target {dn[2:].mean():.1f} dB")
        fig, ((a, b), (c, d)) = plt.subplots(2, 2, figsize=(7.2, H), sharey="row", gridspec_kw=dict(  # 0.5 / H = 0.5 in
            height_ratios=[1.15, 1], left=0.095, right=0.695, top=1 - 0.62 / H, bottom=0.5 / H, hspace=0.73 / 1.475,
            wspace=0.25))
        for ax, (field, ts, spot, lo, hi, num) in zip((c, d), (r["tr_focus"][0], r["opt_focus"][0])):  # emitter 1
            im = ax.imshow(field.T, extent=[-404, 404, -220 / 44.1, 219 / 44.1], **GRAY)
            ax.text(0.03, 0.04, f"median away from target ${num[1]:.1f}$ dB", transform=ax.transAxes, fontsize=8.5,
                    color=INK, bbox=BOX)
            ins = ax.inset_axes((0.03, 0.62, 0.30, 0.35), xticks=[], yticks=[])
            ins.imshow(spot.T, extent=[x[j - 5] - 4, x[j + 5] + 4, ts[0], ts[-1]], **GRAY)
            ins.contour(x[j - 5:j + 6], ts, spot.T, levels=[-3], colors="white", linewidths=0.8, linestyles="dashed")
            ins.plot([x[j], x[j]], [lo, hi], color="white", lw=0.8, solid_capstyle="butt")
            ins.text(0.5, 0.95, f"{num[0]:.0f} $\\mu$s", transform=ins.transAxes, ha="center", va="top", fontsize=7,
                     color=INK, bbox=BOX | dict(boxstyle="round,pad=0.1"))
            ind = ax.indicate_inset_zoom(ins, edgecolor="#111111", lw=0.6, alpha=1)   # thin dark frames, white halo
            plt.setp([ind.rectangle, *ins.spines.values()], path_effects=[pe.withStroke(linewidth=1.6, foreground="w")])
            plt.setp(ind.connectors, lw=0.5, alpha=0.6)
            ax.plot(x[j], ts[0], "^", color=MEAS[0], ms=8, mec="white", mew=0.6,          # below the zoom box
                    transform=mt.offset_copy(ax.transData, fig=fig, y=-7, units="points"))
        ph = fig.add_axes([0.735, 0.245, 0.255, 0.51], xticks=[], yticks=[])
        ph.imshow(plt.imread("data/photo_setup.jpg"))
        for text, xyt, xy in (("loudspeakers", (330, 400), (375, 812)), ("microphone", (1150, 1083), (858, 1217)),
                              ("moving stage", (560, 1895), (1000, 1667))):
            ph.annotate(text, xy, xyt, fontsize=8, ha="center", va="center",
                        bbox=BOX | dict(boxstyle="round,pad=0.18", alpha=0.9), arrowprops=dict(arrowstyle="-", lw=0.8,
                        color="#1A1A1A", shrinkB=1, path_effects=[pe.withStroke(linewidth=2, foreground="w")]))
        ph.text(-0.06, 1.0, "(e)", transform=ph.transAxes, fontsize=11, family="cmb10", va="bottom")
        fig.colorbar(im, cax=fig.add_axes([0.7554, 0.5 / H, 0.2142, 0.016]), orientation="horizontal",
                     ticks=range(-40, 1, 10), label="Response level (dB)").outline.set_linewidth(0.5)
    else:
        r["sup"] = crosstalk(compress(npz["sup_rec"]), pairs)
        print(f"Fig. 3 sup - opt: mean crosstalk {r['sup'][1] - r['opt'][1]} dB")
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, H), sharey=True, gridspec_kw=dict(
            left=0.09, right=0.985, top=1 - 0.576 / H, bottom=0.512 / H, wspace=0.23))
        for e in range(2):                                                     # superposed filters, odd receivers
            b.plot(x[1::2], r["sup"][0][e][1::2], "s", ms=2.3, mfc="none", mec=MEAS[e], mew=0.5, zorder=2.5)
        keys["superposed filters"] = plt.Line2D([], [], ls="", marker="s", ms=2.3, mfc="none", mec="#52514e", mew=0.5)
    a.set_ylim(min(r["opt"][0].min(), r["opt_pred"][0].min()) - 1.6, 1)
    for ax, s, dy, yl in zip(fig.axes, "abcd", (5, 5, 2, 2), ("Crosstalk (dB)", "", "Time relative to peak (ms)", "")):
        ax.text(-0.1 if yl else -0.03, 1.04, f"({s})", transform=ax.transAxes, fontsize=11, family="cmb10", va="bottom")
        ax.set(xlim=(-400, 400), xticks=range(-400, 401, 200), xlabel="Receiver position x (mm)", ylabel=yl)
        ax.yaxis.set_major_locator(plt.MultipleLocator(dy))
        ax.spines[["top", "right"]].set_visible(False)
    for ax, m, xys in zip((a, b), ("tr", "opt"), labels):     # "Emitter i" label positions
        for e, xy in enumerate(xys):
            ax.text(*xy, f"Emitter {e + 1}", fontsize=9, color=MEAS[e], ha="center", va="center", zorder=6, bbox=BOX)
            ax.plot(x, r[m + "_pred"][0][e], color=PRED[e], lw=1.3)
            ax.plot(x[::2], r[m][0][e][::2], "o", ms=2.8, mfc=MEAS[e], mec="white", mew=0.35, zorder=3)
        for t, e in pairs:
            ax.plot(x[t], 0.55, "v", color=MEAS[e], ms=8, zorder=5, clip_on=False)
        ax.grid(color="#e6e5e1", lw=0.6)
    keys["target receiver"] = tuple(plt.Line2D([], [], ls="none", marker="v", color=MEAS[e], ms=6) for e in range(2))
    a.legend(keys.values(), keys, loc="lower right", handler_map={tuple: lh.HandlerTuple(ndivide=None, pad=0.35)},
             fontsize=8.5, handlelength=2.2, borderaxespad=0.6, labelspacing=0.25, edgecolor="#c9c8c3", framealpha=1)
    lvl = r["tr"][1].mean(), r["opt"][1].mean()                    # mean crosstalk as dashed lines, joined by an arrow
    pa, pb = a.get_position(), b.get_position()
    xm, T = (pa.x1 + pb.x0) / 2, mt.blended_transform_factory(fig.transFigure, a.transData)   # x: figure, y: crosstalk
    for ax, p, far, l, title in zip((a, b), (pa, pb), (pa.x0, pb.x1), lvl, ("Corrected time reversal", "Optimized")):
        ax.plot([far, xm], [l, l], transform=T, color="#8a8a8a", lw=0.7, ls=(0, (2.5, 2)), zorder=1, clip_on=False)
        fig.add_artist(plt.Rectangle((p.x0, p.y1 + 0.295 / H), p.width, 0.2 / H, fc="#e9e8e4", ec="none"))  # grey band
        fig.text((p.x0 + p.x1) / 2, p.y1 + 0.395 / H, title, ha="center", va="center", fontsize=11)
    b.annotate("", (xm, lvl[1]), (xm, lvl[0]), xycoords=T,
               arrowprops=dict(arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK, shrinkA=0, shrinkB=0))
    fig.text(xm - 0.004, sum(lvl) / 2, f"$\\approx${10 ** ((lvl[0] - lvl[1]) / 10):.0f}$\\times$ lower", transform=T,
             rotation=90, ha="right", va="center", fontsize=8.5, color=INK)
    fig.savefig(f"fig{k}.pdf", dpi=300)
