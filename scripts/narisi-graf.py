import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path


def safe_latex(text : str) -> str:
    return text.replace("{", "\\{").replace("}", "\\}").replace("_", "\\_")

def safe_latex_param(text : str) -> str:
    return text.replace("_", "\\_")


def format_dec(val: float, decimals: int = 2) -> str:
    return f"{val:.{decimals}f}".replace(".", ",")

def x_labela(params : dict, x_p_ef : bool, x_v : bool) -> str:
    if x_p_ef:
        parallel = params.get("parallel", "null")
        config = params.get("config", {})
        hnsw_ef = config.get("hnsw_ef", "null")
        return f"p={parallel}-ef{hnsw_ef}"
    elif x_v:
        config = params.get("config", {})
        quant = config.get("quantization") or {}
        oversampling = quant.get("oversampling", "null")
        return f"v{oversampling}"
    
def x_naslov(params : dict, x_p_ef : bool, x_v : bool) -> str:
    if x_p_ef:
        parallel = params.get("parallel", "N/A")
        config = params.get("config", {})
        hnsw_ef = config.get("hnsw_ef", "N/A")
        return f"{{p={parallel},\\\\ef={hnsw_ef}}}"
    elif x_v:
        config = params.get("config", {})
        quant = config.get("quantization") or {}
        oversampling = quant.get("oversampling", "Brez")
        return f"{{v={oversampling}}}"
    

def build_latex_pgfplot(data: list, y_metrika: str = "rps", x_p_ef : bool = True, x_v : bool = False) -> str:
    tikz_lines = [
        "\\begin{figure}[htbp]",
        "  \\centering",
        "  \\begin{tikzpicture}",
        "    \\begin{axis}[",
        "      title={TODO: Spremeni me},",
        "      ylabel={" + safe_latex_param(y_metrika) + "},",
        "      symbolic x coords={",
    ]
    
    x_labels = []
    x_oznake = []
    plot_series = {}

    for entry in data:
        run_name = entry.get("name", "Unknown").replace("_", "\\_")
        searches = entry.get("searches", [])

        # Odstrani poizvedbe ki nimajo p=8 in ef=512 v primeru poizvedbe po vzporednih klicih
        if x_v:
            searches[:] = [
                s for s in searches
                if s.get("params", {}).get("config", {}).get("hnsw_ef") == 512
                and s.get("params", {}).get("parallel") == 8
            ]
            
        if x_p_ef:
            searches[:] = [
                            s for s in searches
                            if s.get("params", {}).get("config", {}).get("quantization", {}).get("oversampling") == 2.0
                        ]

        # Sort searches inside the run
        searches_sorted = sorted(
            searches,
            key=lambda s: (
                s.get("params", {}).get("parallel", 0),
                s.get("params", {}).get("config", {}).get("hnsw_ef", 0),
                (s.get("params", {}).get("config", {}).get("quantization") or {}).get("oversampling") or -1.0
            )
        )

        coords = []
        for s in searches_sorted:
            s_params = s.get("params", {})
            s_results = s.get("results", {})
            lbl = x_labela(s_params, x_p_ef=x_p_ef, x_v=x_v)
            x_info=x_naslov(s_params, x_p_ef=x_p_ef, x_v=x_v)

            # Vrednosti x osi
            if lbl not in x_labels:
                x_labels.append(lbl)
                x_oznake.append(x_info)

            # Vrednosti na grafu
            y_val = s_results.get(y_metrika, 0.0)
            coords.append((lbl, y_val))

        plot_series[run_name] = coords

    symbolic_str = ",\n        ".join([f"{lbl}" for lbl in x_labels])
    tikz_lines.append(f"        {symbolic_str}")
    tikz_lines.append("      },")
    
    tikz_lines.append("      xticklabels={")
    oznake_str = ",\n        ".join([f"{oznaka}" for oznaka in x_oznake])
    tikz_lines.append(f"        {oznake_str}")
    tikz_lines.append("      },")
    
    tikz_lines.extend([
        "      xtick=data,",
        "      x tick label style={rotate=45, anchor=east, align=center, font=\\footnotesize},",
        "      /pgf/number format/use comma,",
        "      /pgf/number format/fixed,",
        "      /pgf/number format/precision=3,"
        "      grid=both,",
        "      grid style={line width=.1pt, draw=gray!20},",
        "      major grid style={line width=.2pt, draw=gray!50},",
        "      legend style={at={(0.5, -0.30)}, anchor=north},",
        "      width=0.95\\textwidth,",
        "      height=0.55\\textwidth,",
        "    ]"
    ])
    
    colors = ["blue","red","teal","orange","purple","green","magenta","brown","olive","cyan","crimson","gold"]

    for idx, (series_name, coords) in enumerate(plot_series.items()):
        color = colors[idx % len(colors)]
        tikz_lines.append(f"\n    % Series: {series_name}")
        # OPOMBA: grafično podobo se lahko še malo izboljša
        tikz_lines.append(f"    \\addplot[color={color}, mark=*, thick] coordinates {{")
        
        for x_lbl, y_val in coords:
            tikz_lines.append(f"      ({x_lbl}, {y_val:.4f})")
            
        tikz_lines.append("    };")
        tikz_lines.append(f"    \\addlegendentry{{{series_name}}}")

    tikz_lines.extend([
        "    \\end{axis}",
        "  \\end{tikzpicture}",
        "  \\caption{Primerjava " + safe_latex_param(y_metrika) + " across configurations.}",
        "  \\label{fig:benchmark_todo_" + y_metrika + "}",
        "\\end{figure}"
    ])

    return "\n".join(tikz_lines)


def get_nested_val(data_dict: dict, path_str: str):
    """Dynamically resolves a dot-separated path like 'results.mean_precisions'."""
    keys = path_str.split(".")
    curr = data_dict
    for k in keys:
        if isinstance(curr, dict) and k in curr:
            curr = curr[k]
        else:
            return None
    return curr


def main():
    parser = argparse.ArgumentParser(
        description="Nariši graf na podlagi rezultatov JSON."
    )
    parser.add_argument(
        "--vhod",
        "-f",
        type=str,
        default="./scripts/nabrani-rezultati.json",
        help="Ime JSON datoteke z združenimi rezultati.",
    )
    parser.add_argument(
        "--imena",
        "-n",
        required=True,
        help="Imena testov, razmejena z vejicami."
    )
    parser.add_argument(
        "--x-p-ef",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        help="Uporabljaj kombinacije p/ef vrednosti za x os, privzeta izbira.",
    )
    parser.add_argument(
        "--x-v",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=False,
        help="Uporabljaj vrednosti v (vzorčenja) pri p 8 ef 512 za graf, izklopi gradnjo p-ef grafa.",
    )
    parser.add_argument(
        "--y-vrednost",
        "-y",
        type=str,
        default="mean_time",
        help="Želene metrike (npr. mean_time)",
    )
    parser.add_argument(
        "--izhod",
        "-o",
        type=str,
        default="./scripts/graf.tex",
        help="Datoteka za shranjenje vnosa.",
    )

    def indeks_od(test):
        run_name = test.get("name", "")
        for index, name in enumerate(imena_testov):
            if name in run_name:
                return index
        return len(imena_testov)

    args = parser.parse_args()
    imena_testov = str(args.imena).split(",")

    with open(args.vhod, "r") as f:
        vsi_testi = json.load(f)

    testi = [
        run for run in vsi_testi if any(name in run["name"] for name in imena_testov)
    ]

    testi.sort(key=indeks_od)

    if not len(testi) == len(imena_testov):
        print(f"Niso bili najdeni vsi testi: {imena_testov}, {[test["name"] for test in testi]}")
        return

    latex_code = build_latex_pgfplot(testi, y_metrika=args.y_vrednost, x_p_ef=args.x_p_ef and not args.x_v, x_v=args.x_v)

    output_tex_file = args.izhod
    with open(output_tex_file, "w") as f:
        f.write(latex_code)

    print(f"Ustvaril graf: {output_tex_file}")

if __name__ == "__main__":
    main()