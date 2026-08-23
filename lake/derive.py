"""E5: derive the lake's queryable columns from a captured workflow blob.

Everything here is a pure function of (workflow JSON, reference tables). That is
the whole point of the raw-bucket design — if a column turns out wrong, or a new
one is wanted, it is recomputed from the bucket instead of recrawled.

Reference tables, all snapshot-versioned so a rebuild is reproducible:

* `core_nodes.json` — core class list + hidden-PROMPT consumers + input names
  (from `probes/core_nodes.py`, stamped with the ComfyUI commit it was read at);
* `extension-node-map.json` — ComfyUI-Manager's repo -> class_type index, used
  for pack attribution when a node carries no `cnr_id` stamp.

Two workflow formats are handled and normalised to the same columns:

* **save** — the litegraph canvas graph (`nodes`/`links`), what the frontend
  saves and what `workflow` PNG chunks carry;
* **api**  — the executor's prompt graph (`{id: {class_type, inputs}}`), what
  `prompt` chunks carry.

The save format is strictly richer (it has per-node version stamps and the
frontend version; the API format has neither), so where both exist the save
format wins and the API graph is kept only as corroboration.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from collections import Counter
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

# Loader inputs whose binding time we care about: literal in the graph (the
# workflow names a file) vs fed by a link (the file is chosen at run time by an
# upstream node). Flagship query #1.
LOADER_INPUTS = (
    "ckpt_name",
    "unet_name",
    "lora_name",
    "vae_name",
    "clip_name",
    "control_net_name",
    "style_model_name",
    "gguf_name",
    "model_name",
    "diffusion_model",
)


def _norm_repo(url: str) -> str:
    """`https://github.com/Owner/Name.git/` -> `github.com/owner/name`."""
    if not url:
        return ""
    url = re.sub(r"^https?://", "", url.strip().rstrip("/"))
    url = re.sub(r"^www\.", "", url)
    return re.sub(r"\.git$", "", url).lower()


MODEL_EDGE_TYPES = {"MODEL"}

# Litegraph's own virtual nodes. They appear in every save-format graph, they
# are not Python nodes, so they are in neither the core class list nor any
# pack's node map — and left alone they dominate the "unattributable" count
# (Note/MarkdownNote/Reroute/PrimitiveNode were 5,417 of the sample's node
# instances). They are core in the sense that matters here: nothing installs them.
FRONTEND_NATIVE = {
    "Note",
    "MarkdownNote",
    "Reroute",
    "PrimitiveNode",
    "PrimitiveBoolean",
    "PrimitiveFloat",
    "PrimitiveInt",
    "PrimitiveString",
    "PrimitiveStringMultiline",
}


@dataclass
class Reference:
    core: set[str]
    hidden_prompt: set[str]
    core_inputs: dict[str, list[str]]
    node_to_packs: dict[str, list[str]]
    pack_titles: dict[str, str]
    comfyui_commit: str = ""
    # normalised repo -> canonical pack id, so a pack attributed by name and the
    # same pack attributed by `cnr_id` stamp do not become two different packs
    repo_to_pack: dict[str, str] = field(default_factory=dict)

    def canonical_pack(self, ident: str) -> str:
        return self.repo_to_pack.get(_norm_repo(ident), ident)

    @classmethod
    def load(
        cls,
        core_path: str | pathlib.Path | None = None,
        map_path: str | pathlib.Path | None = None,
        packs_path: str | pathlib.Path = "data/packs.jsonl",
    ) -> Reference:
        if core_path is None:
            # Packaged, not cwd-relative: a console script does not put the cwd
            # on sys.path, so a relative default silently resolves against
            # whatever directory the process happened to start in.
            core_doc = json.loads(files("lake").joinpath("reference/core_nodes.json").read_text())
        else:
            core = pathlib.Path(core_path)
            if not core.exists():
                # The experiments repo regenerates it into data/.
                alt = pathlib.Path("data/core_nodes.json")
                core = alt if alt.exists() else core
            core_doc = json.loads(core.read_text())
        node_to_packs: dict[str, list[str]] = {}
        pack_titles: dict[str, str] = {}
        if map_path is None:
            # Packaged for the same reason as the core list: the previous default
            # was a path under $HOME that exists on a developer machine and in no
            # container. `lake selftest` is what surfaced it.
            raw = json.loads(
                files("lake").joinpath("reference/extension-node-map.json").read_text()
            )
        else:
            raw = json.loads(pathlib.Path(map_path).read_text())
        for repo, value in raw.items():
            nodes = value[0] if isinstance(value, list) and value else []
            meta = value[1] if isinstance(value, list) and len(value) > 1 else {}
            if isinstance(meta, dict) and meta.get("title_aux"):
                pack_titles[repo] = meta["title_aux"]
            for name in nodes:
                node_to_packs.setdefault(name, []).append(repo)
        repo_to_pack: dict[str, str] = {}
        packs_file = pathlib.Path(packs_path)
        if packs_file.exists():
            for line in packs_file.open():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                repo = _norm_repo(row.get("repo") or row.get("repo_url") or "")
                if repo and row.get("pack_id"):
                    # A registered pack wins the key: its cnr_id is what stamps use.
                    if repo not in repo_to_pack or row.get("cnr_id"):
                        repo_to_pack[repo] = row["pack_id"]
        return cls(
            core=set(core_doc["class_types"]) | FRONTEND_NATIVE,
            hidden_prompt=set(core_doc.get("hidden_prompt_class_types") or []),
            core_inputs=core_doc.get("inputs") or {},
            node_to_packs=node_to_packs,
            pack_titles=pack_titles,
            comfyui_commit=core_doc.get("commit", ""),
            repo_to_pack=repo_to_pack,
        )


@dataclass
class NodeRow:
    """One row of the `workflow_nodes` table."""

    node_id: str
    class_type: str
    is_core: bool
    cnr_id: str | None = None
    ver: str | None = None
    aux_id: str | None = None
    mode: int | None = None
    packs: list[str] = field(default_factory=list)
    pack_ambiguous: bool = False
    pack_unknown: bool = False


@dataclass
class Derived:
    """One row of the `workflows` table, plus its child rows."""

    format: str
    node_count: int = 0
    link_count: int = 0
    frontend_version: str | None = None
    graph_version: Any = None
    model_edges: int = 0
    hidden_prompt_nodes: int = 0
    group_count: int = 0
    muted_or_bypassed: int = 0
    nodes: list[NodeRow] = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)
    bindings: list[dict] = field(default_factory=list)
    pack_set: list[str] = field(default_factory=list)
    stamped_nodes: int = 0
    cnr_nodes: int = 0
    aux_nodes: int = 0
    canonical_hash: str = ""
    structural_hash: str = ""
    exact_hash: str = ""
    error: str | None = None


def derive(obj: Any, ref: Reference) -> Derived:
    kind = _classify(obj)
    if kind == "save":
        out = _derive_save(obj, ref)
    elif kind == "api":
        out = _derive_api(obj, ref)
    else:
        return Derived(format="unknown", error="unrecognised graph shape")
    out.exact_hash = hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    out.canonical_hash = canonical_hash(obj, kind)
    out.structural_hash = canonical_hash(obj, kind, keep_widget_values=False)
    out.pack_set = sorted({p for n in out.nodes for p in n.packs})
    out.stamped_nodes = sum(1 for n in out.nodes if n.cnr_id or n.aux_id)
    out.cnr_nodes = sum(1 for n in out.nodes if n.cnr_id)
    out.aux_nodes = sum(1 for n in out.nodes if n.aux_id)
    return out


def _classify(obj: Any) -> str:
    if not isinstance(obj, dict):
        return "unknown"
    if isinstance(obj.get("nodes"), list):
        return "save"
    values = [v for v in obj.values() if isinstance(v, dict)]
    if values and sum(1 for v in values if "class_type" in v) >= max(1, len(values) // 2):
        return "api"
    return "unknown"


def _attribute(class_type: str, cnr_id: str | None, ref: Reference) -> tuple[list[str], bool, bool]:
    """(packs, ambiguous, unknown) for one node.

    A `cnr_id` stamp written by the frontend is authoritative and collapses the
    ambiguity entirely; the name-map lookup is the fallback for unstamped nodes.
    """
    if cnr_id and cnr_id != "comfy-core":
        return [ref.canonical_pack(cnr_id)], False, False
    if class_type in ref.core:
        return [], False, False
    packs = [ref.canonical_pack(p) for p in (ref.node_to_packs.get(class_type) or [])]
    packs = sorted(set(packs))
    return packs, len(packs) > 1, not packs


def _derive_save(obj: dict, ref: Reference) -> Derived:
    out = Derived(format="save")
    extra = obj.get("extra") or {}
    out.frontend_version = extra.get("frontendVersion")
    out.graph_version = obj.get("version")
    out.group_count = len(obj.get("groups") or [])

    nodes = obj.get("nodes") or []
    out.node_count = len(nodes)
    for node in nodes:
        if not isinstance(node, dict):
            continue
        class_type = node.get("type") or ""
        props = node.get("properties") or {}
        cnr_id = props.get("cnr_id")
        packs, ambiguous, unknown = _attribute(class_type, cnr_id, ref)
        out.nodes.append(
            NodeRow(
                node_id=str(node.get("id")),
                class_type=class_type,
                is_core=class_type in ref.core,
                cnr_id=cnr_id,
                ver=props.get("ver"),
                aux_id=props.get("aux_id"),
                mode=node.get("mode"),
                packs=packs,
                pack_ambiguous=ambiguous,
                pack_unknown=unknown,
            )
        )
        out.class_counts[class_type] += 1
        if node.get("mode") in (2, 4):  # muted / bypassed
            out.muted_or_bypassed += 1
        if class_type in ref.hidden_prompt:
            out.hidden_prompt_nodes += 1
        out.bindings.extend(_save_bindings(node, class_type, ref, str(node.get("id"))))

    links = obj.get("links") or []
    out.link_count = len(links)
    for link in links:
        # [link_id, origin_node, origin_slot, target_node, target_slot, type]
        if (
            isinstance(link, list)
            and len(link) >= 6
            and link[5] in MODEL_EDGE_TYPES
            or isinstance(link, dict)
            and link.get("type") in MODEL_EDGE_TYPES
        ):
            out.model_edges += 1
    return out


def _save_bindings(node: dict, class_type: str, ref: Reference, node_id: str) -> list[dict]:
    """Literal-vs-link for each loader input this node declares.

    In the save format a widget that has been fed by a link appears in `inputs[]`
    carrying a `widget` descriptor; a widget still holding a literal does not
    appear there at all. So the input's presence in `inputs[]` *is* the answer —
    but only for inputs we know the node has, which is why the core input table
    is needed.
    """
    declared = ref.core_inputs.get(class_type)
    linked: dict[str, bool] = {}
    for inp in node.get("inputs") or []:
        if not isinstance(inp, dict):
            continue
        widget = inp.get("widget")
        name = (widget or {}).get("name") if isinstance(widget, dict) else None
        name = name or inp.get("name")
        if name in LOADER_INPUTS:
            linked[name] = inp.get("link") is not None
    rows = []
    seen = set()
    for name, is_linked in linked.items():
        seen.add(name)
        rows.append(
            {
                "node_id": node_id,
                "class_type": class_type,
                "input": name,
                "binding": "link" if is_linked else "literal",
                "source": "save.inputs",
            }
        )
    if declared:
        for name in declared:
            if name in LOADER_INPUTS and name not in seen:
                rows.append(
                    {
                        "node_id": node_id,
                        "class_type": class_type,
                        "input": name,
                        "binding": "literal",
                        "source": "save.widget",
                    }
                )
    return rows


def _derive_api(obj: dict, ref: Reference) -> Derived:
    out = Derived(format="api")
    out.node_count = len(obj)
    for node_id, node in obj.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type") or ""
        packs, ambiguous, unknown = _attribute(class_type, None, ref)
        out.nodes.append(
            NodeRow(
                node_id=str(node_id),
                class_type=class_type,
                is_core=class_type in ref.core,
                packs=packs,
                pack_ambiguous=ambiguous,
                pack_unknown=unknown,
            )
        )
        out.class_counts[class_type] += 1
        if class_type in ref.hidden_prompt:
            out.hidden_prompt_nodes += 1
        for name, value in (node.get("inputs") or {}).items():
            is_link = isinstance(value, list) and len(value) == 2
            if is_link:
                out.link_count += 1
            if name in LOADER_INPUTS:
                out.bindings.append(
                    {
                        "node_id": str(node_id),
                        "class_type": class_type,
                        "input": name,
                        "binding": "link" if is_link else "literal",
                        "source": "api.inputs",
                    }
                )
    # The API graph carries no edge types, so MODEL edges are inferred from the
    # upstream node's declared outputs — approximated by the input name `model`.
    for node in obj.values():
        if not isinstance(node, dict):
            continue
        for name, value in (node.get("inputs") or {}).items():
            if name in ("model", "unet") and isinstance(value, list):
                out.model_edges += 1
    return out


# --- canonicalisation (E4) --------------------------------------------------


def canonical_form(obj: Any, kind: str | None = None, *, keep_widget_values: bool = True) -> Any:
    """Layout-free, id-free structural form of a graph.

    Strips everything a re-share mutates without changing the workflow: node ids,
    canvas positions/sizes, execution order, colours, groups, the frontend
    version stamp. What survives is the multiset of (class_type, widget values)
    and the multiset of typed edges rewritten in terms of those nodes.

    `keep_widget_values=False` drops the parameters too, leaving pure topology.
    That is the level at which "the same workflow, re-run with another seed"
    collapses — which is most of what a gallery of generated images contains.
    """
    kind = kind or _classify(obj)
    if kind == "save":
        nodes = obj.get("nodes") or []
        keyed = []
        for node in nodes:
            if not isinstance(node, dict):
                continue
            values = (
                json.dumps(node.get("widgets_values"), sort_keys=True, default=str)
                if keep_widget_values
                else ""
            )
            keyed.append((node.get("type") or "", values, node.get("id")))
        order = sorted(range(len(keyed)), key=lambda i: (keyed[i][0], keyed[i][1]))
        rank = {keyed[i][2]: pos for pos, i in enumerate(order)}
        canon_nodes = [[keyed[i][0], keyed[i][1]] for i in order]
        edges = []
        for link in obj.get("links") or []:
            if isinstance(link, list) and len(link) >= 6:
                src, ssl, dst, dsl, etype = link[1], link[2], link[3], link[4], link[5]
            elif isinstance(link, dict):
                src, ssl = link.get("origin_id"), link.get("origin_slot")
                dst, dsl = link.get("target_id"), link.get("target_slot")
                etype = link.get("type")
            else:
                continue
            edges.append([rank.get(src, -1), ssl, rank.get(dst, -1), dsl, etype])
        return {"n": canon_nodes, "e": sorted(edges, key=json.dumps)}
    if kind == "api":
        keyed = []
        for node_id, node in (obj or {}).items():
            if not isinstance(node, dict):
                continue
            literals = {
                k: v for k, v in (node.get("inputs") or {}).items() if not isinstance(v, list)
            }
            values = (
                json.dumps(literals, sort_keys=True, default=str)
                if keep_widget_values
                else json.dumps(sorted(literals), default=str)
            )
            keyed.append((node.get("class_type") or "", values, node_id))
        order = sorted(range(len(keyed)), key=lambda i: (keyed[i][0], keyed[i][1]))
        rank = {keyed[i][2]: pos for pos, i in enumerate(order)}
        canon_nodes = [[keyed[i][0], keyed[i][1]] for i in order]
        edges = []
        for node_id, node in (obj or {}).items():
            if not isinstance(node, dict):
                continue
            for name, value in (node.get("inputs") or {}).items():
                if isinstance(value, list) and len(value) == 2:
                    edges.append(
                        [
                            rank.get(str(value[0]), rank.get(value[0], -1)),
                            value[1],
                            rank.get(node_id, -1),
                            name,
                        ]
                    )
        return {"n": canon_nodes, "e": sorted(edges, key=json.dumps)}
    return obj


def canonical_hash(obj: Any, kind: str | None = None, *, keep_widget_values: bool = True) -> str:
    form = canonical_form(obj, kind, keep_widget_values=keep_widget_values)
    return hashlib.sha256(
        json.dumps(form, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
