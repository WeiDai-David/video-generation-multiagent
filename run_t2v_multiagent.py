# -*- coding: utf-8 -*-
"""
多智能体 -> 面向大一新生的“逐帧”的文生视频 Prompt 生成
依赖:
    pip install dashscope requests
"""

import os, re, json, time, uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ==========================
# 路径初始化
# ==========================
DATA_DIR = Path("history"); DATA_DIR.mkdir(exist_ok=True)
MEM_DIR = Path("memory"); MEM_DIR.mkdir(exist_ok=True)
MEM_FILE = MEM_DIR / "knowledge.json"
if not MEM_FILE.exists():
    MEM_FILE.write_text(json.dumps({
        "banned_logos": [],
        "style": {"palette": "blue-white", "subtitle": {"font": "Noto Sans SC", "size": 36}},
        "jargon_blacklist": ["范畴论","鞅","变分下界","伴随算子","谱半径"],
        "freshman_ok_terms": ["注意力","查询","键","值","权重","热力图","多头"],
        "math_templates": {"scaled_dot": "softmax(QK^T/\\sqrt{d})V"}
    }, ensure_ascii=False, indent=2), encoding="utf-8")

# ==========================
# LLM 客户端
# ==========================
class LLMClient:
    def complete(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        raise NotImplementedError

class QwenClient(LLMClient):
    """
    使用 dashscope 新版 SDK 的稳定写法。
    - 可选 region: "intl" | "cn"
    - 默认模型改为 qwen-plus
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-plus", region: str = "intl"):
        import dashscope
        self.api_key = (api_key or os.getenv("DASHSCOPE_API_KEY", "").strip())
        if not self.api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")
        self.model = model
        self.region = region

        # 设定 API Key 与域名
        dashscope.api_key = self.api_key
        if self.region == "intl":
            dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
        else:
            dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"

    def _extract_json(self, text: str) -> str:
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.I)
        if m: return m.group(1)
        m = re.search(r"(\{[\s\S]*\})", text)
        if m: return m.group(1)
        return text.strip()

    def complete(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.2, max_tokens: int = 2000,
                 ensure_json: bool = False, **kwargs) -> str:
        try:
            from dashscope import Generation
            resp = Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                result_format="message"
            )

            # ✅ 新版 SDK 输出兼容解析
            text = None

            # (1) 如果直接有 output_text 属性
            if "output_text" in resp:
                text = resp["output_text"]

            # (2) 否则尝试新版结构
            elif "output" in resp and isinstance(resp["output"], dict):
                output = resp["output"]
                if "choices" in output and len(output["choices"]) > 0:
                    choice = output["choices"][0]
                    msg = choice.get("message", {})
                    text = msg.get("content") or msg.get("text")

            # (3) 最后兜底
            if not text:
                text = str(resp)

            if not text.strip():
                raise RuntimeError(f"Qwen 返回空输出: {resp}")

        except Exception as e:
            msg = str(e)
            if "401" in msg or "Unauthorized" in msg:
                raise RuntimeError(
                    "HTTP 401：请检查 API Key 是否有效、region 是否为 intl，以及模型（推荐 qwen-plus）。") from e
            raise

        return self._extract_json(text) if ensure_json else text


# ==========================
# 简洁度 & 逐帧校验
# ==========================
FORBIDDEN_VAGUE = {"更酷","优雅地","震撼","华丽","自然地","随机地"}
ALLOWED_ACTIONS = {
    "appear","disappear",
    "sweep",            # {"direction":"left-to-right"|"right-to-left"|"top-to-bottom"|...}
    "highlight",        # {"mode":"word-by-word"|"range", "order":"left-to-right"|"by-index"}
    "emphasis",         # {"type":"pulse"|"scale"|"glow"}
    "caption_set",      # {"text":"..."}
    "opacity_to"        # {"a": 0~1}
}
ALLOWED_POSITIONS = {"left","center","right","top","bottom","top-left","top-right","bottom-left","bottom-right"}
MAX_STEPS_PER_SCENE = 12
MAX_ACTIONS_PER_STEP = 4
MAX_FORMULAS_PER_SCENE = 1  # 大一友好：最多1处公式
MAX_JARGON_PER_SCENE = 4

def contains_coordinates(obj: Any) -> bool:
    """不允许出现 bbox/x/y/坐标字段。"""
    if isinstance(obj, dict):
        for k,v in obj.items():
            if k.lower() in {"x","y","w","h","bbox","start_bbox","position_xy"}:
                return True
            if contains_coordinates(v):
                return True
    elif isinstance(obj, list):
        return any(contains_coordinates(x) for x in obj)
    return False

def count_formulas(scene: Dict[str,Any]) -> int:
    s = json.dumps(scene, ensure_ascii=False)
    return len(re.findall(r"\\sqrt|\\frac|\\sum|\\int|\\begin\{matrix\}|\\left|\\right|\\cdot|\\times|\\alpha|\\beta|softmax|QK\^T", s))

def count_jargon(scene: Dict[str,Any]) -> int:
    kb = json.loads(MEM_FILE.read_text(encoding="utf-8"))
    bad = kb.get("jargon_blacklist", [])
    s = json.dumps(scene, ensure_ascii=False)
    return sum(s.count(w) for w in bad)

def validate_scene_simple(scene: Dict[str,Any]) -> Tuple[bool, List[str]]:
    issues = []
    # 1) 禁止坐标
    if contains_coordinates(scene):
        issues.append("出现坐标/框字段（请改用 position/sweep/highlight 的方位或顺序描述）")
    # 2) 帧步与动作数量限制
    steps = scene.get("frame_steps", [])
    if len(steps) == 0:
        issues.append("缺少 frame_steps")
    if len(steps) > MAX_STEPS_PER_SCENE:
        issues.append(f"帧步过多：{len(steps)}（建议 ≤ {MAX_STEPS_PER_SCENE}）")
    for i, st in enumerate(steps):
        acts = st.get("actions", [])
        if len(acts) == 0:
            issues.append(f"step#{i} 缺少 actions")
        if len(acts) > MAX_ACTIONS_PER_STEP:
            issues.append(f"step#{i} 动作过多：{len(acts)}（建议 ≤ {MAX_ACTIONS_PER_STEP}）")
        # 动作合法性检查 & 文本空泛词
        for a in acts:
            # 操作名
            keys = set(a.keys()) - {"actor"}  # e.g. {"highlight": {...}}
            if not keys:
                issues.append(f"step#{i} 存在无效动作（缺少原子指令）")
                continue
            op = list(keys)[0]
            if op not in ALLOWED_ACTIONS:
                issues.append(f"step#{i} 包含不支持的动作：{op}")
        cap = st.get("caption","") or ""
        if any(v in cap for v in FORBIDDEN_VAGUE):
            issues.append(f"step#{i} 含空泛词：{cap}")
    # 3) 公式/术语密度
    fcnt = count_formulas(scene)
    if fcnt > MAX_FORMULAS_PER_SCENE:
        issues.append(f"公式过多：{fcnt}（建议 ≤ {MAX_FORMULAS_PER_SCENE}）")
    jcnt = count_jargon(scene)
    if jcnt > MAX_JARGON_PER_SCENE:
        issues.append(f"高阶术语过多：{jcnt}（建议 ≤ {MAX_JARGON_PER_SCENE}）")
    return (len(issues) == 0), issues

# ==========================
# 4 个最小 Agent
# ==========================
class OutlineAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, topic: str, audience: str, total_sec: int, segments: int = 6) -> Dict[str, Any]:
        sys = "你是资深科普视频编导。仅输出 JSON。面向大一新生，语言简洁、比喻清晰。"
        user = f"""
任务：为主题“{topic}”生成视频大纲（{segments} 段，总时长 {total_sec}s）。
受众：{audience}（大一新生）
要求：每段给标题、目标、3–4条要点、预计时长；避免高阶术语和过多公式。

仅输出 JSON：
{{
  "topic": "{topic}",
  "target_audience": "{audience}",
  "total_duration_sec": {total_sec},
  "segments": [
    {{
      "id": 1, "title": "引入", "goal": "建立动机",
      "bullets": ["问题","示例","提出概念"], "est_time_sec": {max(10, total_sec//segments)}
    }}
  ]
}}
""".strip()
        txt = self.llm.complete(sys, user, ensure_json=True)
        data = json.loads(txt)
        segs = data.get("segments", [])
        if len(segs) < segments:
            base = max(10, total_sec // segments)
            for i in range(len(segs), segments):
                segs.append({
                    "id": i+1, "title": f"第{i+1}段",
                    "goal": "讲解要点",
                    "bullets": ["要点1","要点2","要点3"],
                    "est_time_sec": base
                })
            data["segments"] = segs
        return data

class ExpansionAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        sys = "你是“分镜+文生视频提示工程”专家。仅输出 JSON。不要使用坐标；用方位和顺序描述。"
        user = f"""
将片段扩写为“逐步可执行 JSON”（面向大一新生），**禁止坐标/数值**，用“方位/方向/顺序”描述：
- 必须包含 actors[]：每个有 id、type、position（从 {sorted(list(ALLOWED_POSITIONS))} 里选或简单短语如 "right-half"）
- 必须包含 frame_steps[]：每步含 t（起始秒）、duration_sec、actions[]
- 原子动作仅限：{sorted(list(ALLOWED_ACTIONS))}
  - 例：{{"actor":"sent","sweep":{{"direction":"left-to-right"}}}}
       {{\"actor\":\"sent\",\"highlight\":{{\"mode\":\"word-by-word\",\"order\":\"left-to-right\"}}}}
       {{\"actor\":\"note\",\"caption_set\":{{\"text\":\"要点1\"}}}}
       {{\"actor\":\"halo\",\"emphasis\":{{\"type\":\"pulse\"}}}}
- 禁止抽象词（更酷/优雅地…），避免专业黑话；公式最多 1 处。
- 每步 ≤ {MAX_ACTIONS_PER_STEP} 个动作；每段总步数 ≤ {MAX_STEPS_PER_SCENE}。
- 旁白要短句化、口语化。

片段：
{json.dumps(segment, ensure_ascii=False)}

输出 JSON 模板：
{{
  "id": {segment.get("id", 1)},
  "title": "{segment.get("title","")}",
  "narration": "旁白全文…",
  "actors": [
    {{"id":"sent","type":"text_line","position":"center"}}
  ],
  "frame_steps": [
    {{"t":0.0,"duration_sec":0.6,"actions":[{{"actor":"sent","appear":true}}],"caption":"句子出现"}},
    {{"t":0.6,"duration_sec":0.6,"actions":[{{"actor":"sent","highlight":{{"mode":"word-by-word","order":"left-to-right"}}}}],"caption":"从左到右依次高亮"}}
  ],
  "transitions":"cut",
  "duration_sec": {segment.get("est_time_sec", 20)}
}}
仅输出 JSON！
""".strip()
        txt = self.llm.complete(sys, user, ensure_json=True)
        scene = json.loads(txt)
        ok, issues = validate_scene_simple(scene)
        scene.setdefault("_validation", {})["ok"] = ok
        scene["_validation"]["issues"] = issues
        return scene

class FrameClarityCritic:
    """逐帧 + 简洁度审查（低复杂度、低术语/公式密度）"""
    def run(self, scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        issues = []
        for sc in scenes:
            ok = sc.get("_validation",{}).get("ok", False)
            if not ok:
                for it in sc.get("_validation",{}).get("issues", []):
                    issues.append(f"段落{sc.get('id')}: {it}")
            # 进一步：标题/旁白长度检查
            nar = (sc.get("narration") or "")
            if len(nar) > 260:  # 过长旁白（约 >20s）
                issues.append(f"段落{sc.get('id')}: 旁白过长（建议分句/删减）")
        return {
            "critic": "FrameClarityCritic",
            "scores": {
                "framewise": 5 if not issues else 3,
                "simplicity": 5 if not issues else 3
            },
            "issues": issues,
            "global_suggestions": [
                "禁止坐标，用 position/sweep/highlight 描述方位与顺序",
                "每步≤4个动作、每段≤12步，旁白短句化"
            ]
        }

class PromptEditor:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm
    def run(self, scenes: List[Dict[str, Any]], total_sec: int = 150) -> Dict[str, Any]:
        return {
            "global": {
                "language": "zh-CN",
                "tone": "科技感 + 教学动画 + 通俗",
                "fps": 24,
                "color_theme": "blue-white",
                "music": "light ambient, no vocals",
                "subtitles": True,
                "safety": {"no_brand_logos": True}
            },
            "timeline": scenes
        }

# ==========================
# Orchestrator
# ==========================
class Orchestrator:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.outline_agent = OutlineAgent(llm)
        self.expander = ExpansionAgent(llm)
        self.critic = FrameClarityCritic()
        self.editor = PromptEditor()

    def run_once(self, topic: str, audience: str, total_sec: int = 150) -> Dict[str, Any]:
        folder = DATA_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        folder.mkdir(parents=True, exist_ok=True)

        outline = self.outline_agent.run(topic, audience, total_sec)
        (folder / "outline.json").write_text(json.dumps(outline, ensure_ascii=False, indent=2), encoding="utf-8")

        scenes = [self.expander.run(seg) for seg in outline["segments"]]
        (folder / "expanded.json").write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")

        critique = self.critic.run(scenes)
        (folder / "critique.json").write_text(json.dumps(critique, ensure_ascii=False, indent=2), encoding="utf-8")

        final_prompt = self.editor.run(scenes, total_sec=total_sec)
        (folder / "final_prompt.json").write_text(json.dumps(final_prompt, ensure_ascii=False, indent=2), encoding="utf-8")

        print("[OK] 生成完成！结果保存在：", folder)
        print("[提示] 把 final_prompt.json 贴到你的文生视频平台。")
        return {"folder": str(folder), "final_prompt": final_prompt, "critique": critique}

# ==========================
# 入口
# ==========================
if __name__ == "__main__":
    # 直接在代码里写 Key（仅本地调试）
    # os.environ["DASHSCOPE_API_KEY"] = "你的真实APIkey"

    llm = QwenClient(
        api_key="api_key",
        model="qwen-plus",
        region="intl"
    )
    orch = Orchestrator(llm)
    orch.run_once(topic="注意力机制", audience="工科大一", total_sec=150)
