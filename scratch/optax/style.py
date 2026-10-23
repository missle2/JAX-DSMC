"""Names, colours and line styles of the optimiser comparison (2026-10-23; the centroid-fixed optcm arms added
the same day).  Colours: the
dataviz reference categorical palette in its fixed order (validated: CVD and normal-vision
separation pass; the light slots need the legend / labels the figures carry)."""

LABELS = {
    "ox_adam": "Adam",
    "adam": "Adam (2nd run)",
    "ox_adamcos": "Adam + cosine step",
    "ox_ademamix": "AdEMAMix",
    "ox_sfadam": "Schedule-free Adam",
    "ox_prodigy": "Prodigy",
    "tr": "Trust region",
    "lbfgs": "L-BFGS",
    "mma": "MMA",
    "ox_adam_cm": "Adam, centroid fixed",
    "ox_adamcos_cm": "Adam + cosine, centroid fixed",
    "ox_prodigy_cm": "Prodigy, centroid fixed",
    "ox_prodigycos_cm": "Prodigy + cosine cap, centroid fixed",
}
COLOURS = {
    "ox_adam": "#2a78d6",
    "adam": "#2a78d6",
    "ox_adamcos": "#eb6834",
    "ox_ademamix": "#1baf7a",
    "ox_sfadam": "#eda100",
    "ox_prodigy": "#e87ba4",
    "tr": "#008300",
    "lbfgs": "#6250d6",
    "mma": "#e34948",
    # the centroid-fixed arms (optcm, 2026-10-23): their free arm's hue, dashed; Prodigy with
    # the cosine cap has no free run and takes Prodigy's hue dash-dotted
    "ox_adam_cm": "#2a78d6",
    "ox_adamcos_cm": "#eb6834",
    "ox_prodigy_cm": "#e87ba4",
    "ox_prodigycos_cm": "#e87ba4",
}
DASHED = {"adam", "ox_adam_cm", "ox_adamcos_cm", "ox_prodigy_cm"}
DASHDOT = {"ox_prodigycos_cm"}
FREE = {"ox_adam_cm": "ox_adam", "ox_adamcos_cm": "ox_adamcos", "ox_prodigy_cm": "ox_prodigy"}


def linestyle(arm):
    return "-." if arm in DASHDOT else "--" if arm in DASHED else "-"


# the comparison's twelve arms (the first, pre-optax Adam run, "adam", dropped 2026-10-23: it
# duplicates "ox_adam"); the centroid-fixed four last
ORDER = ("ox_adam", "ox_adamcos", "ox_ademamix", "ox_sfadam", "ox_prodigy", "tr", "lbfgs", "mma",
         "ox_adam_cm", "ox_adamcos_cm", "ox_prodigy_cm", "ox_prodigycos_cm")  # fmt: skip
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REGIME_TITLES = {"kn0.5": "Kn 0.5, Mach 0.3", "kn0.5_M10": "Kn 0.5, Mach 10"}
