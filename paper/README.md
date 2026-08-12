# Paper branch workspace

The default branch keeps paper repositories outside the runtime engine. Long-lived
`paper/p0x-*` branches are isolated execution workspaces for one paper at a time.

Each paper branch:

- contains the complete PHM-Vibench engine;
- contains exactly one paper overlay under `paper/project/`;
- keeps runnable paper configs under `configs/experiments/`;
- has no gitlinks to sibling paper repositories.

The corresponding PaperTrace repository pins this branch through `src/vibench`.
Claims, manuscripts, submission material, and evidence ledgers remain authoritative
in that PaperTrace repository.
