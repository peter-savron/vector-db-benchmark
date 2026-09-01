import json
import locale
from pathlib import Path

rezultati = Path("./rezultati")
imena_testov_pot = Path("./scripts/imena-testov.json")
output_file = Path("./scripts/nabrani-rezultati.json")
latex_tabele_path = Path("./scripts/latex-tabele.tex")

def safe_latex(text : str) -> str:
    return text.replace("{", "\\{").replace("}", "\\}").replace("_", "\\_")

def safe_latex_param(text : str) -> str:
    return text.replace("_", "\\_")

def format_decimal(num : float, dec_pos: int = 2) -> str:
    tekst = f"{num:,.{dec_pos}f}"
    
    return tekst.replace(",", "X").replace(".", ",").replace("X", ".")

def testni_parametri(params: dict) -> str:
    parallel = params.get("parallel")
    config = params.get("config", {})
    hnsw_ef = config.get("hnsw_ef")

    quantization = config.get("quantization") or {}
    oversampling = quantization.get("oversampling")

    parts = []
    
    parts.append(f"\\makecell{{\\texttt{{p}}: {parallel}")
    parts.append(f"\\\\\\texttt{{ef}}: {hnsw_ef}")
        
    if oversampling is not None:
        # Format integer/float values nicely (e.g., 32.0 -> 32)
        val_str = f"{format_decimal(oversampling)}" if isinstance(oversampling, float) else str(oversampling)
        parts.append(f"\\\\\\texttt{{v}}: {val_str}")
        
    parts.append("}")

    return "".join(parts)

folders = [
    str(folder.relative_to(rezultati)) 
    for folder in rezultati.rglob("*") 
    if folder.is_dir() and not any(child.is_dir() for child in folder.iterdir())
]
aggregated_data = []

with open(imena_testov_pot, "w") as f:
    json.dump(folders, f, indent=4)

print(f"Našel {len(folders)} zagonov in shranil v {imena_testov_pot}")

for folder_str in folders:
    folder_path = rezultati / folder_str

    upload_files = list(folder_path.glob("*upload*.json"))
    search_files = list(folder_path.glob("*search*.json"))

    if not upload_files or not search_files:
        print(f"{folder_path} is empty or partially filled.")
        continue

    if len(upload_files) != 1:
        print(f"Subfolder '{folder_str}' contains {len(upload_files)} build files (expected 1). Skipping folder.")
        continue

    with open(upload_files[0], "r") as f:
        build_content = json.load(f)

    searches_content = []
    for search_file in search_files:
        with open(search_file, "r") as f:
            searches_content.append(json.load(f))

    aggregated_data.append({
        "name": f"{folder_str}/{upload_files[0].name}",
        "build": build_content,
        "searches": searches_content
    })

## Popravi agregirane rezultate, odstrani začetno poizvedbo (p=42
for entry in aggregated_data:
    entry["searches"] = [
        s for s in entry.get("searches", [])
        if not (
            s.get("params", {}).get("parallel") == 42
        )
    ]

    entry["searches"].sort(
        key=lambda s: (
            s.get("params", {}).get("parallel", 0),
            s.get("params", {}).get("config", {}).get("hnsw_ef", 0),
            (s.get("params", {}).get("config", {}).get("quantization") or {}).get("oversampling") or -1.0
        )
    )

with open(output_file, "w") as f:
    json.dump(aggregated_data, f, indent=4)

latex_output = []

for entry in aggregated_data:
    folder_name = entry["name"]
    build_info = entry.get("build", {})
    build_params = build_info.get("params", {})
    build_results = build_info.get("results", {})
    searches = entry.get("searches", [])

    hnsw_podatki = build_params.get("hnsw_config", "N/A")
    kvantizacija = build_params.get("quantization_config", "N/A")
    tovor = build_params.get("payload_index_params", "N/A")
    upload_time = build_results.get("upload_time", 0.0)
    cas_indeksiranja = build_results.get("total_time", 0.0) - upload_time

    # Construct the table caption containing upload/build data
    caption = (
        f"Test: \\texttt{{{safe_latex_param(folder_name)}}} | "
        f"Nastavitve HNSW kazala: \\texttt{{{safe_latex_param(str(hnsw_podatki))}}} | "
        f"Nastavitve kvantizacije: \\texttt{{{safe_latex_param(str(kvantizacija))}}} | "
        f"Nastavitve kazala tovora: \\texttt{{{safe_latex_param(str(tovor))}}} | "
        f"Čas nalaganja: {format_decimal(upload_time)} s Okviren čas gradnje kazala: {format_decimal(cas_indeksiranja)} s"
    )

    # Label key sanitized for LaTeX references
    label_key = folder_name.replace("/", ":")

    # Build LaTeX table
    table_lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "\\small",
        f"  \\caption{{{safe_latex_param(folder_name)}}}",
        f"  \\label{{tab:{label_key}}}",
        "  \\begin{tabularx}{\\textwidth}{Xrrrrrr}",
        "    \\toprule",
        "    Nastavitve poizvedbe & Priklic & \\makecell{Povprečna\\\\latenca (s)} & \\makecell{Standarden\\\\odklon (s)} & \\makecell{Prepustnost\\\\(RPS)} & \\makecell{p95 latenca\\\\(s)}  \\\\",
        "    \\midrule"
    ]

    # Populate rows for each search run
    for s in searches:
        s_params = s.get("params", {})
        s_res = s.get("results", {})

        if s_params.get("parallel") == 42:
            continue

        hnsw_ef = s_params.get("config", "N/A")
        priklic = s_res.get("mean_precisions", 0.0)
        povprecen_cas = s_res.get("mean_time", 0.0)
        odklon_cas = s_res.get("std_time", 0.0)
        propustnost = s_res.get("rps", 0.0)
        p95_time = s_res.get("p95_time", 0.0)
        p99_time = s_res.get("p99_time", 0.0)

        row = (
            f"    {testni_parametri(s_params)} & "
            f"{format_decimal(priklic, 4)} & "
            f"{format_decimal(povprecen_cas, 6)} & "
            f"{format_decimal(odklon_cas, 6)} & "
            f"{format_decimal(propustnost)} & "
            f"{format_decimal(p95_time, 6)} \\\\"
        )
        table_lines.append(row)

    table_lines.extend([
        "    \\bottomrule",
        "  \\end{tabularx}",
        f"\\caption* {{{caption}}}"
        "\\end{table}\n"
    ])

    latex_output.append("\n".join(table_lines))

with open(latex_tabele_path, "w") as f:
    f.write("\n\n".join(latex_output))

print(f"Ustvarjenih {len(aggregated_data)} LaTeX tabel, zapisane so v  {latex_tabele_path}")
