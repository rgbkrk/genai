"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("genai").rglob("*.py")):
    module_path = path.relative_to("genai").with_suffix("")
    doc_path = path.relative_to("genai").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    parts = list(module_path.parts)

    if parts[-1].startswith("_"): 
        continue
    
    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)
        print("::: " + identifier)
    # break

# nav["mkdocs_autorefs", "references"] = "autorefs/references.md"
# nav["mkdocs_autorefs", "plugin"] = "autorefs/plugin.md"

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:  # 
    nav_file.writelines(nav.build_literate_nav())
