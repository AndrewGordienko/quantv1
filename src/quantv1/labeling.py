"""Minimal localhost human-labelling interface for the MGRM gold set.

Data tooling only -- it does NOT touch the model, extractor, research harness,
promotion gates, or certification architecture. It renders each frozen source
filing next to a labelling form, saves drafts, validates numeric consistency,
and exports authoritative JSONL deterministically.

Integrity: development documents may show the separate, non-authoritative
extractor prefill in a clearly marked panel; certification documents NEVER show
prefill or any extractor output. Human labels are authoritative.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT
from .ingest.guidance import ACTIONS, ALLOWED_METRICS, STATUSES
from .ingest import mgrm_corpus


DRAFT_DIR = mgrm_corpus.GOLDSET_DIR / "drafts"
DEV_LABELS_PATH = mgrm_corpus.GOLDSET_DIR / "mgrm_dev_labels.jsonl"
CERT_LABELS_PATH = mgrm_corpus.GOLDSET_DIR / "mgrm_cert_labels.jsonl"
UNITS = ["absolute", "per_share", "percent"]
_NUMERIC_REQUIRED_STATUS = {"AVAILABLE", "REAFFIRMED"}


def _manifest_index() -> dict[str, dict]:
    return {record["document_id"]: record for record in mgrm_corpus.load_manifest()}


def _prefill_index() -> dict[str, list]:
    path = mgrm_corpus.DEV_PREFILL_PATH
    index: dict[str, list] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                record = json.loads(line)
                index[record["document_id"]] = record.get("suggested_records", [])
    return index


def documents(split: str | None = None) -> list[dict]:
    records = mgrm_corpus.load_manifest()
    drafts = {p.stem for p in DRAFT_DIR.glob("*.json")} if DRAFT_DIR.exists() else set()
    out = []
    for record in records:
        if split and record["split"] != split:
            continue
        out.append({"document_id": record["document_id"],
                    "company": record["company"], "sector": record["sector"],
                    "format": record["format_hint"], "split": record["split"],
                    "labelled": record["document_id"] in drafts})
    return out


def document_view(doc_id: str) -> dict:
    record = _manifest_index().get(doc_id)
    if record is None:
        raise KeyError(doc_id)
    view = {"document_id": doc_id, "company": record["company"],
            "sector": record["sector"], "format": record["format_hint"],
            "split": record["split"], "source_url": record["document_url"],
            "document_type": record["document_type"], "prefill": None}
    # Prefill is DEVELOPMENT-ONLY. Certification documents never expose it.
    if record["split"] == "development":
        view["prefill"] = _prefill_index().get(doc_id, [])
    return view


def source_html(doc_id: str) -> str:
    record = _manifest_index().get(doc_id)
    if record is None:
        raise KeyError(doc_id)
    path = ROOT / record["raw_path"]
    if not path.exists():
        return ("<p>Source not present locally. Run "
                "<code>scripts/mgrm_corpus.py rehydrate</code>.</p>")
    return path.read_text(encoding="utf-8", errors="replace")


def _as_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return "INVALID"


def validate_label(payload: dict) -> list[str]:
    errors = []
    no_guidance = payload.get("no_guidance")
    expected = payload.get("expected", []) or []
    if no_guidance not in (True, False):
        errors.append("no_guidance must be true or false")
    if no_guidance is True and expected:
        errors.append("no_guidance=true requires an empty expected list")
    if no_guidance is False and not expected:
        errors.append("guidance present requires at least one expected record")
    for index, record in enumerate(expected):
        tag = f"record {index + 1}"
        for field in ("metric", "period", "units", "status", "action", "evidence"):
            if not record.get(field):
                errors.append(f"{tag}: missing {field}")
        if record.get("metric") and record["metric"] not in ALLOWED_METRICS:
            errors.append(f"{tag}: metric not in {sorted(ALLOWED_METRICS)}")
        if record.get("units") and record["units"] not in UNITS:
            errors.append(f"{tag}: units must be one of {UNITS}")
        if record.get("status") and record["status"] not in STATUSES:
            errors.append(f"{tag}: status not in {sorted(STATUSES)}")
        if record.get("action") and record["action"] not in ACTIONS:
            errors.append(f"{tag}: action not in {sorted(ACTIONS)}")
        low, high, mid = (_as_float(record.get(k)) for k in ("low", "high", "midpoint"))
        if "INVALID" in (low, high, mid):
            errors.append(f"{tag}: low/high/midpoint must be numeric")
            continue
        numeric_required = record.get("status") in _NUMERIC_REQUIRED_STATUS
        if numeric_required and None in (low, high, mid):
            errors.append(f"{tag}: {record.get('status')} guidance needs low, high, midpoint")
        if None not in (low, high, mid):
            if not (low <= mid <= high):
                errors.append(f"{tag}: require low <= midpoint <= high")
            scale = max(abs(low), abs(high), 1.0)
            if abs(mid - (low + high) / 2) > 0.01 * scale:
                errors.append(f"{tag}: midpoint must be ~ (low+high)/2")
    return errors


def save_draft(doc_id: str, payload: dict) -> dict:
    if doc_id not in _manifest_index():
        raise KeyError(doc_id)
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    errors = validate_label(payload)
    (DRAFT_DIR / f"{doc_id}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"saved": True, "valid": not errors, "errors": errors}


def load_draft(doc_id: str) -> dict | None:
    path = DRAFT_DIR / f"{doc_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _label_record(doc_id: str, manifest: dict, draft: dict) -> dict:
    expected = [] if draft.get("no_guidance") else [
        {"metric": r.get("metric"), "period": r.get("period"),
         "units": r.get("units"), "currency": r.get("currency") or None,
         "low": _as_float(r.get("low")), "high": _as_float(r.get("high")),
         "midpoint": _as_float(r.get("midpoint")), "status": r.get("status"),
         "action": r.get("action"), "evidence": r.get("evidence")}
        for r in draft.get("expected", [])]
    return {"doc_id": doc_id, "company": manifest["company"],
            "sector": manifest["sector"], "format": manifest["format_hint"],
            "source_url": manifest["document_url"], "split": manifest["split"],
            "no_guidance": bool(draft.get("no_guidance")), "expected": expected}


def export_jsonl(split: str) -> dict:
    """Deterministically export validated drafts for a split to authoritative JSONL."""
    manifest = _manifest_index()
    records, skipped = [], []
    for doc_id in sorted(manifest):
        record = manifest[doc_id]
        if record["split"] != split:
            continue
        draft = load_draft(doc_id)
        if draft is None:
            skipped.append({"document_id": doc_id, "reason": "UNLABELLED"})
            continue
        errors = validate_label(draft)
        if errors:
            skipped.append({"document_id": doc_id, "reason": "INVALID",
                            "errors": errors})
            continue
        records.append(_label_record(doc_id, record, draft))
    path = DEV_LABELS_PATH if split == "development" else CERT_LABELS_PATH
    header = (f"# MGRM {split} AUTHORITATIVE human labels (exported deterministically).\n"
              f"# {len(records)} labelled; append certification labels to the gold "
              "set only when the extractor is frozen.\n")
    lines = [json.dumps(record, sort_keys=True, default=str) for record in records]
    path.write_text(header + "\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8")
    return {"split": split, "exported": len(records), "skipped": skipped,
            "path": str(path.relative_to(ROOT))}


# ---- HTTP layer (FastAPI, own port; does not touch the dashboard API) --------

def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    app = FastAPI(title="MGRM labelling")

    class Draft(BaseModel):
        no_guidance: bool | None = None
        expected: list[dict] = []

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _render_page()

    @app.get("/api/documents")
    def api_documents(split: str | None = None):
        return documents(split)

    @app.get("/api/document/{doc_id}")
    def api_document(doc_id: str):
        try:
            view = document_view(doc_id)
        except KeyError:
            raise HTTPException(404, "unknown document")
        view["draft"] = load_draft(doc_id)
        return view

    @app.get("/api/source/{doc_id}", response_class=HTMLResponse)
    def api_source(doc_id: str):
        try:
            return source_html(doc_id)
        except KeyError:
            raise HTTPException(404, "unknown document")

    @app.put("/api/draft/{doc_id}")
    def api_save(doc_id: str, draft: Draft):
        try:
            return save_draft(doc_id, draft.model_dump())
        except KeyError:
            raise HTTPException(404, "unknown document")

    @app.post("/api/export")
    def api_export(split: str):
        return JSONResponse(export_jsonl(split))

    return app


_PAGE = """<!doctype html><html><head><meta charset=utf-8><title>MGRM labelling</title>
<style>
 body{margin:0;font:14px system-ui;display:flex;height:100vh}
 #list{width:220px;overflow:auto;border-right:1px solid #ccc;padding:8px}
 #list div{padding:4px;cursor:pointer;border-radius:4px}
 #list div.done{color:#0a0}#list div.sel{background:#def}
 #src{flex:1;border:0}#form{width:420px;overflow:auto;padding:10px;border-left:1px solid #ccc}
 .rec{border:1px solid #ddd;padding:6px;margin:6px 0;border-radius:4px}
 label{display:block;font-size:12px;color:#555;margin-top:4px}
 input,select,textarea{width:100%;box-sizing:border-box}
 #prefill{background:#fff8e1;border:1px solid #e0c060;padding:6px;font-size:12px;white-space:pre-wrap}
 #err{color:#b00;font-size:12px;white-space:pre-wrap}button{margin:2px 0}
 .tag{font-size:11px;padding:1px 5px;border-radius:3px;background:#eee}
</style></head><body>
<div id=list></div>
<iframe id=src sandbox></iframe>
<div id=form>
 <div><b id=title>select a document</b> <span id=split class=tag></span></div>
 <label>Guidance present?</label>
 <select id=ng><option value="">--</option><option value="no">No (no-guidance)</option>
  <option value="yes">Yes</option></select>
 <div id=records></div>
 <button onclick=addRec()>+ add record</button>
 <div id=prefillWrap style=display:none><b>DEV prefill (NON-AUTHORITATIVE hint)</b>
  <div id=prefill></div></div>
 <div><button onclick=save()>Save draft</button>
  <button onclick="exportSplit()">Export split JSONL</button></div>
 <div id=err></div>
</div>
<script>
const METRICS=%METRICS%,STATUSES=%STATUSES%,ACTIONS=%ACTIONS%,UNITS=%UNITS%;
let docs=[],cur=null;
async function boot(){docs=await (await fetch('/api/documents')).json();render()}
function render(){const l=document.getElementById('list');l.innerHTML='';
 docs.forEach(d=>{const e=document.createElement('div');
  e.textContent=(d.labelled?'✓ ':'• ')+d.company+' ['+d.sector.slice(0,4)+'] '+d.split[0];
  e.className=(d.labelled?'done ':'')+(cur===d.document_id?'sel':'');
  e.onclick=()=>open(d.document_id);l.appendChild(e)})}
function opt(v,sel){return '<option'+(v===sel?' selected':'')+'>'+v+'</option>'}
function recHtml(r){r=r||{};return '<div class=rec>'
 +'<label>metric</label><select class=m><option></option>'+METRICS.map(x=>opt(x,r.metric)).join('')+'</select>'
 +'<label>period (e.g. FY2024, Q4-2024)</label><input class=p value="'+(r.period||'')+'">'
 +'<label>units</label><select class=u><option></option>'+UNITS.map(x=>opt(x,r.units)).join('')+'</select>'
 +'<label>currency</label><input class=c value="'+(r.currency||'')+'">'
 +'<label>low</label><input class=lo value="'+(r.low??'')+'">'
 +'<label>high</label><input class=hi value="'+(r.high??'')+'">'
 +'<label>midpoint</label><input class=mi value="'+(r.midpoint??'')+'">'
 +'<label>status</label><select class=s><option></option>'+STATUSES.map(x=>opt(x,r.status)).join('')+'</select>'
 +'<label>action</label><select class=a><option></option>'+ACTIONS.map(x=>opt(x,r.action)).join('')+'</select>'
 +'<label>evidence (exact sentence or table cells)</label><textarea class=e>'+(r.evidence||'')+'</textarea>'
 +'<button onclick="this.parentNode.remove()">remove</button></div>'}
function addRec(r){document.getElementById('records').insertAdjacentHTML('beforeend',recHtml(r))}
async function open(id){cur=id;document.getElementById('err').textContent='';
 const v=await (await fetch('/api/document/'+id)).json();
 document.getElementById('title').textContent=v.company+' — '+v.document_type;
 document.getElementById('split').textContent=v.split;
 document.getElementById('src').src='/api/source/'+id;
 document.getElementById('records').innerHTML='';
 const d=v.draft;document.getElementById('ng').value=d?(d.no_guidance?'no':'yes'):'';
 if(d&&!d.no_guidance)(d.expected||[]).forEach(addRec);
 const pw=document.getElementById('prefillWrap');
 if(v.prefill){pw.style.display='block';
  document.getElementById('prefill').textContent=JSON.stringify(v.prefill,null,1)}
 else pw.style.display='none';render()}
function collect(){const ng=document.getElementById('ng').value;
 const recs=[...document.querySelectorAll('.rec')].map(x=>({
  metric:x.querySelector('.m').value,period:x.querySelector('.p').value,
  units:x.querySelector('.u').value,currency:x.querySelector('.c').value,
  low:x.querySelector('.lo').value,high:x.querySelector('.hi').value,
  midpoint:x.querySelector('.mi').value,status:x.querySelector('.s').value,
  action:x.querySelector('.a').value,evidence:x.querySelector('.e').value}));
 return {no_guidance:ng==='no'?true:ng==='yes'?false:null,expected:ng==='no'?[]:recs}}
async function save(){if(!cur)return;
 const r=await (await fetch('/api/draft/'+cur,{method:'PUT',headers:{'Content-Type':'application/json'},
  body:JSON.stringify(collect())})).json();
 document.getElementById('err').textContent=r.valid?'saved (valid)':'saved. FIX:\\n'+r.errors.join('\\n');
 const d=docs.find(x=>x.document_id===cur);if(d)d.labelled=true;render()}
async function exportSplit(){if(!cur)return;const sp=docs.find(x=>x.document_id===cur).split;
 const r=await (await fetch('/api/export?split='+sp,{method:'POST'})).json();
 document.getElementById('err').textContent='exported '+r.exported+' to '+r.path
  +'\\nskipped '+r.skipped.length}
boot();
</script></body></html>"""


def _render_page() -> str:
    return (_PAGE.replace("%METRICS%", json.dumps(sorted(ALLOWED_METRICS)))
            .replace("%STATUSES%", json.dumps(sorted(STATUSES)))
            .replace("%ACTIONS%", json.dumps(sorted(ACTIONS)))
            .replace("%UNITS%", json.dumps(UNITS)))
