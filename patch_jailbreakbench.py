"""Patch the installed `jailbreakbench` package to register modern target models.

The published jailbreakbench package (1.0.0) only knows four target models
(Vicuna-13B, Llama-2-7B, GPT-3.5-Turbo-1106, GPT-4-0125-preview) in its internal
`Model` enum. Running PAIR against a modern target (GPT-4o) fails inside the
package, regardless of this repo's own config.py.

This script inserts the extra models into the four structures jailbreakbench needs
(Model enum, API_KEYS, LITELLM_MODEL_NAMES, SYSTEM_PROMPTS). It is idempotent at the
line level: each insertion is skipped if already present, so it is safe to re-run and
to extend with more models. Re-run it after any `pip install/upgrade jailbreakbench`
(which overwrites the package files).

Usage:
    python patch_jailbreakbench.py
"""

import os

import jailbreakbench

CONFIG_PATH = os.path.join(os.path.dirname(jailbreakbench.__file__), "config.py")

# Each tuple is (anchor_line, line_to_insert_after_anchor). Anchors must be unique
# substrings of the target file. Insertions are applied only if not already present.
PATCHES = [
    # --- GPT-4o (OpenAI, snapshot gpt-4o-2024-11-20) ---
    (
        '    gpt_4 = "gpt-4-0125-preview"',
        '    gpt_4o = "gpt-4o-2024-11-20"',
    ),
    (
        '    Model.gpt_4: "OPENAI_API_KEY",',
        '    Model.gpt_4o: "OPENAI_API_KEY",',
    ),
    (
        '    Model.gpt_4: "gpt-4-0125-preview",',
        '    Model.gpt_4o: "gpt-4o-2024-11-20",',
    ),
    (
        "    Model.gpt_4: None,",
        "    Model.gpt_4o: None,",
    ),
]


def main():
    with open(CONFIG_PATH, "r") as f:
        source = f.read()

    added = []
    for anchor, insertion in PATCHES:
        if insertion in source:
            continue  # already patched
        if anchor not in source:
            raise RuntimeError(
                f"Anchor not found in {CONFIG_PATH}:\n  {anchor!r}\n"
                "The jailbreakbench version may differ from the one this patch "
                "was written for (1.0.0). Inspect config.py and update PATCHES."
            )
        source = source.replace(anchor, f"{anchor}\n{insertion}", 1)
        added.append(insertion.strip())

    if not added:
        print(f"Already patched (nothing to do): {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, "w") as f:
        f.write(source)

    print(f"Patched {CONFIG_PATH}. Added {len(added)} line(s):")
    for line in added:
        print(f"  + {line}")


if __name__ == "__main__":
    main()
