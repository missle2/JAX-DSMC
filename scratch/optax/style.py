"""Names, colours and line styles of the optimiser comparison (2026-10-23).  Colours: the
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
}
DASHED = {"adam"}
ORDER = ("ox_adam", "adam", "ox_adamcos", "ox_ademamix", "ox_sfadam", "ox_prodigy", "tr",
         "lbfgs", "mma")  # fmt: skip
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REGIME_TITLES = {"kn0.5": "Kn 0.5, Mach 0.3", "kn0.5_M10": "Kn 0.5, Mach 10"}
