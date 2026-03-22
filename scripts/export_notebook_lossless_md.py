import argparse
import json
from pathlib import Path


def join_source(source):
    if isinstance(source, list):
        return "".join(source)
    return source or ""


def fenced(code: str, lang: str = "") -> str:
    return f"```{lang}\n{code}\n```\n"


def to_lossless_markdown(nb_path: Path, out_path: Path) -> None:
    raw_nb_text = nb_path.read_text(encoding="utf-8")
    nb = json.loads(raw_nb_text)

    lines = []
    lines.append(f"# {nb_path.stem} (Lossless Markdown Export)\n")
    lines.append(
        "This file preserves all notebook information and also renders markdown cells directly, including Mermaid diagrams.\n"
    )

    # Full JSON snapshot guarantees no information loss.
    lines.append("## Raw Notebook JSON\n")
    lines.append(
        "<details>\n<summary>Show full notebook JSON (lossless)</summary>\n\n"
    )
    lines.append(
        fenced(raw_nb_text, "json")
    )
    lines.append("</details>\n")

    lines.append("## Rendered Cells\n")
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        cell_type = cell.get("cell_type", "unknown")
        lines.append(f"### Cell {idx} ({cell_type})\n")

        lines.append("<details>\n<summary>Show raw cell JSON</summary>\n\n")
        lines.append(fenced(json.dumps(cell, ensure_ascii=False, indent=2), "json"))
        lines.append("</details>\n")

        source = join_source(cell.get("source", []))
        if cell_type == "markdown":
            # Keep markdown as markdown so Mermaid blocks render in preview.
            lines.append(source if source.endswith("\n") else source + "\n")
            lines.append("\n")
        elif cell_type == "code":
            lang = cell.get("metadata", {}).get("language", "python")
            lines.append(fenced(source, lang))
            execution_count = cell.get("execution_count")
            lines.append(f"Execution count: {execution_count}\n")
            outputs = cell.get("outputs", [])
            lines.append(f"Outputs: {len(outputs)}\n")
            for out_i, output in enumerate(outputs, start=1):
                lines.append(f"#### Output {out_i}\n")
                lines.append(fenced(json.dumps(output, ensure_ascii=False, indent=2), "json"))
        else:
            lines.append("Unsupported cell type for direct rendering. See raw JSON above.\n\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export notebook to lossless markdown.")
    parser.add_argument("input", help="Path to input .ipynb file")
    parser.add_argument("-o", "--output", help="Path to output .md file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Notebook not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_suffix(".md")
    to_lossless_markdown(input_path, output_path)
    print(f"Wrote lossless markdown to: {output_path}")


if __name__ == "__main__":
    main()
