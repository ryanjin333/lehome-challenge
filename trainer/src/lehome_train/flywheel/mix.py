"""Freeze audited 70/30 prepared-dataset inputs before statistics."""
from __future__ import annotations
from dataclasses import dataclass
import json, math, random
from pathlib import Path
from typing import Mapping
from lehome_train.io import canonical_json_sha256, sha256_file

GRADE_WEIGHTS = {"A": 1.0, "B": 0.5}

@dataclass(frozen=True, slots=True)
class MixPlan:
    seed: int; organizer_training_frames: int; flywheel_training_frames: int
    source_weights: dict[str, float]; grade_weights: dict[str, float]
    organizer_episode_ids: tuple[str, ...]; flywheel_episode_ids: tuple[str, ...]
    source_revisions: dict[str, str]; raw_manifest_hashes: tuple[str, ...]; sha256: str
    def body(self): return {"schema_version":1,"seed":self.seed,"organizer_training_frames":self.organizer_training_frames,"flywheel_training_frames":self.flywheel_training_frames,"source_weights":self.source_weights,"grade_weights":self.grade_weights,"organizer_episode_ids":list(self.organizer_episode_ids),"flywheel_episode_ids":list(self.flywheel_episode_ids),"source_revisions":self.source_revisions,"raw_manifest_hashes":list(self.raw_manifest_hashes)}
    def to_dict(self): return self.body() | {"sha256":self.sha256}

def _read(path: Path) -> dict[str, object]:
    try: value=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): raise ValueError(f"mix metadata unavailable: {path}") from None
    if not isinstance(value,dict): raise ValueError("mix metadata must be an object")
    return value
def _ids(manifest: Mapping[str,object]) -> list[str]:
    ids=manifest.get("train_episode_ids")
    if not isinstance(ids,list) or not ids or not all(isinstance(i,str) for i in ids): raise ValueError("mix input has no train episodes")
    return sorted(ids)
def _frames(manifest: Mapping[str,object]) -> int:
    frames=manifest.get("frame_count")
    if type(frames) is not int or frames<=0: raise ValueError("mix input frame count is invalid")
    return frames
def _cycle(ids:list[str], count:int, rng:random.Random)->tuple[str,...]:
    ordered=list(ids); rng.shuffle(ordered); return tuple(ordered[index%len(ordered)] for index in range(count))

def build_mix_plan(organizer: str|Path, flywheel: str|Path, *, seed:int, organizer_fraction:float=.70)->MixPlan:
    if type(seed) is not int or organizer_fraction != .70: raise ValueError("mix requires organizer fraction 0.70")
    org_root, fly_root=Path(organizer),Path(flywheel); org=_read(org_root/"manifest.json"); fly=_read(fly_root/"manifest.json")
    if org.get("output_format")!="groot_lerobot_v2.1_per_episode" or fly.get("output_format")!="groot_lerobot_v2.1_per_episode": raise ValueError("mix inputs must be canonical prepared datasets")
    prov=_read(fly_root/"meta"/"materialization-provenance.json")
    if prov.get("raw_manifest_verified") is not True: raise ValueError("raw artifact was not checksum verified")
    if prov.get("quality_grade") not in GRADE_WEIGHTS: raise ValueError("flywheel grade must be A or B")
    identity=prov.get("raw_identity")
    if not isinstance(identity,Mapping) or identity.get("release_stage")=="public_unseen": raise ValueError("evaluation holdout cannot enter mix")
    raw_hash=prov.get("raw_manifest_sha256")
    if not isinstance(raw_hash,str) or len(raw_hash)!=64: raise ValueError("raw manifest hash is invalid")
    total=max(_frames(org),math.ceil(_frames(fly)/.30)); org_count=round(total*.70); fly_count=total-org_count; rng=random.Random(seed)
    source_revisions={"organizer":str(org.get("source_revision",sha256_file(org_root/"manifest.json"))),"flywheel":sha256_file(fly_root/"manifest.json")}
    body={"schema_version":1,"seed":seed,"organizer_training_frames":org_count,"flywheel_training_frames":fly_count,"source_weights":{"organizer":.7,"flywheel":.3},"grade_weights":GRADE_WEIGHTS,"organizer_episode_ids":list(_cycle(_ids(org),org_count,rng)),"flywheel_episode_ids":list(_cycle(_ids(fly),fly_count,rng)),"source_revisions":source_revisions,"raw_manifest_hashes":[raw_hash]}
    return MixPlan(seed,org_count,fly_count,{"organizer":.7,"flywheel":.3},dict(GRADE_WEIGHTS),tuple(body["organizer_episode_ids"]),tuple(body["flywheel_episode_ids"]),source_revisions,(raw_hash,),canonical_json_sha256(body))
def verify_mix_plan(plan:MixPlan)->None:
    if canonical_json_sha256(plan.body())!=plan.sha256: raise ValueError("frozen mix plan hash is invalid")
